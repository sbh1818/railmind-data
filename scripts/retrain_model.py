"""
RailMind - Biweekly Model Retrainer (GitHub Actions version)
Merges historical + daily data, retrains CatBoost, updates lookup table.
"""
import sys
import os
import json
import time
import glob
import pickle
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (relative to repo root)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_CSV = REPO_ROOT / 'data' / 'historical' / 'daily_delays_flat.csv'
COLLECTED_DIR = REPO_ROOT / 'data' / 'daily_collected'
MODELS_DIR = REPO_ROOT / 'models'
TOP200_FILE = REPO_ROOT / 'data' / 'top200_trains.json'
TRAIN_META_FILE = REPO_ROOT / 'data' / 'train_metadata.json'

RANDOM_STATE = 42
ON_TIME_THRESHOLD = 10
MAX_DELAY_CAP = 720
FOG_MONTHS = [12, 1, 2]
MONSOON_MONTHS = [7, 8, 9]
FESTIVAL_PERIODS = [
    (3, 1, 5), (4, 10, 15), (10, 15, 25), (10, 28, 31),
    (11, 1, 5), (11, 10, 20), (12, 25, 31), (1, 1, 3),
]
TRAIN_TYPES = {
    'RAJDHANI': 0, 'SHATABDI': 1, 'DURONTO': 2, 'VANDE_BHARAT': 3,
    'GARIB_RATH': 4, 'JAN_SHATABDI': 5, 'MAIL_EXPRESS': 6,
    'SUPERFAST': 7, 'OTHER': 8,
}
IMPROVEMENT_THRESHOLD = 0.5


# ============================================================
# FEATURE ENGINEERING (self-contained for Actions)
# ============================================================

def add_time_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_monsoon'] = df['month'].isin(MONSOON_MONTHS).astype(int)
    df['is_fog_season'] = df['month'].isin(FOG_MONTHS).astype(int)
    df['is_festival'] = 0
    for month, day_start, day_end in FESTIVAL_PERIODS:
        mask = (df['month'] == month) & (df['day_of_month'] >= day_start) & (df['day_of_month'] <= day_end)
        df.loc[mask, 'is_festival'] = 1
    df['is_on_time'] = (df['delay_minutes'] <= ON_TIME_THRESHOLD).astype(int)
    return df


def add_metadata(df):
    if TRAIN_META_FILE.exists():
        with open(TRAIN_META_FILE, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta_df = pd.DataFrame([{
            'train_no': str(t.get('train_no', '')).strip(),
            'train_type': t.get('train_type', 'OTHER'),
            'num_stops': t.get('num_stops', 20),
        } for t in meta])
        df = df.merge(meta_df, on='train_no', how='left')
    df['train_type'] = df.get('train_type', pd.Series(['OTHER'] * len(df))).fillna('OTHER')
    df['train_type_encoded'] = df['train_type'].map(TRAIN_TYPES).fillna(8).astype(int)
    if 'num_stops' not in df.columns:
        df['num_stops'] = 20
    df['num_stops'] = df['num_stops'].fillna(20).astype(int)
    return df


def engineer_features(df):
    """Compute all ML features."""
    df = df.sort_values(['train_no', 'date', 'station_code']).reset_index(drop=True)
    
    # Stop sequence
    df['stop_sequence'] = df.groupby(['train_no', 'date']).cumcount() + 1
    
    # Prev station delay (cascade)
    df['prev_station_delay'] = df.groupby(['train_no', 'date'])['delay_minutes'].shift(1).fillna(0)
    
    # Historical averages (expanding, shifted)
    df['train_avg_delay'] = df.groupby('train_no')['delay_minutes'].transform(lambda x: x.expanding().mean().shift(1))
    df['train_std_delay'] = df.groupby('train_no')['delay_minutes'].transform(lambda x: x.expanding().std().shift(1))
    df['station_avg_delay'] = df.groupby('station_code')['delay_minutes'].transform(lambda x: x.expanding().mean().shift(1))
    df['train_station_avg_delay'] = df.groupby(['train_no', 'station_code'])['delay_minutes'].transform(lambda x: x.expanding().mean().shift(1))
    df['train_station_avg_delay'] = df['train_station_avg_delay'].fillna(df['train_avg_delay'])
    df['train_ontime_rate'] = df.groupby('train_no')['is_on_time'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.5)
    
    # Rolling 7-day average
    daily_train = df.groupby(['train_no', 'date'])['delay_minutes'].mean().reset_index()
    daily_train = daily_train.sort_values(['train_no', 'date'])
    daily_train['train_rolling_7d_avg'] = daily_train.groupby('train_no')['delay_minutes'].transform(lambda x: x.rolling(7, min_periods=3).mean().shift(1))
    df = df.merge(daily_train[['train_no', 'date', 'train_rolling_7d_avg']], on=['train_no', 'date'], how='left')
    df['train_rolling_7d_avg'] = df['train_rolling_7d_avg'].fillna(df['train_avg_delay'])
    
    # Consecutive late days
    daily_late = df.groupby(['train_no', 'date'])['delay_minutes'].mean().reset_index()
    daily_late['was_late'] = (daily_late['delay_minutes'] > ON_TIME_THRESHOLD).astype(int)
    daily_late = daily_late.sort_values(['train_no', 'date'])
    def count_consec(series):
        result, count = [], 0
        for val in series:
            result.append(count)
            count = count + 1 if val == 1 else 0
        return result
    daily_late['consecutive_late_days'] = daily_late.groupby('train_no')['was_late'].transform(count_consec)
    df = df.merge(daily_late[['train_no', 'date', 'consecutive_late_days']], on=['train_no', 'date'], how='left')
    df['consecutive_late_days'] = df['consecutive_late_days'].fillna(0)
    
    # Route average
    route_info = df.groupby(['train_no', 'date']).agg(
        first_station=('station_code', 'first'), last_station=('station_code', 'last'),
        route_delay=('delay_minutes', 'mean')).reset_index()
    route_info['route'] = route_info['first_station'] + '_' + route_info['last_station']
    route_info = route_info.sort_values(['route', 'date'])
    route_info['route_avg_delay'] = route_info.groupby('route')['route_delay'].transform(lambda x: x.expanding().mean().shift(1))
    df = df.merge(route_info[['train_no', 'date', 'route_avg_delay']], on=['train_no', 'date'], how='left')
    df['route_avg_delay'] = df['route_avg_delay'].fillna(df['train_avg_delay'])
    
    # Interactions
    df['type_x_fog'] = df['train_type_encoded'] * df['is_fog_season']
    df['type_x_monsoon'] = df['train_type_encoded'] * df['is_monsoon']
    df['type_x_weekend'] = df['train_type_encoded'] * df['is_weekend']
    df['seq_x_train_avg'] = df['stop_sequence'] * df['train_avg_delay'].fillna(0)
    
    # Confidence
    df['train_data_count'] = df.groupby('train_no')['delay_minutes'].transform(lambda x: x.expanding().count().shift(1)).fillna(0)
    
    # Drop rows without history
    df = df.dropna(subset=['train_avg_delay', 'station_avg_delay']).reset_index(drop=True)
    
    return df


FEATURE_COLUMNS = [
    'prev_station_delay', 'train_avg_delay', 'station_avg_delay',
    'train_station_avg_delay', 'train_type_encoded', 'month',
    'is_fog_season', 'stop_sequence', 'num_stops', 'day_of_week',
    'route_avg_delay', 'train_rolling_7d_avg', 'consecutive_late_days',
    'train_std_delay', 'train_ontime_rate', 'is_festival', 'is_weekend',
    'is_monsoon', 'day_of_month', 'type_x_fog', 'type_x_monsoon',
    'type_x_weekend', 'seq_x_train_avg', 'train_data_count',
]


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_retrain():
    logger.info("=" * 60)
    logger.info("RAILMIND - BIWEEKLY MODEL RETRAINER")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    t_start = time.time()
    
    # Step 1: Load and merge data
    logger.info("STEP 1: Loading data...")
    df = pd.read_csv(HISTORICAL_CSV, dtype={'train_no': str, 'train_name': str, 'station_code': str, 'delay_minutes': str}, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['delay_minutes'] = pd.to_numeric(df['delay_minutes'], errors='coerce')
    df = df.dropna(subset=['date', 'delay_minutes'])
    df['delay_minutes'] = df['delay_minutes'].clip(0, MAX_DELAY_CAP).astype(int)
    df['station_code'] = df['station_code'].str.strip().str.upper()
    logger.info(f"  Historical: {len(df):,} records")
    
    # Merge daily collected
    daily_files = sorted(glob.glob(str(COLLECTED_DIR / 'delays_*.csv')))
    logger.info(f"  Found {len(daily_files)} daily files")
    if daily_files:
        daily_dfs = []
        for f in daily_files:
            try:
                dd = pd.read_csv(f)
                aligned = pd.DataFrame({
                    'train_no': dd['train_no'].astype(str),
                    'train_name': dd['train_name'],
                    'date': pd.to_datetime(dd['collection_date']),
                    'station_code': dd['station_code'].str.upper(),
                    'delay_minutes': dd['arrival_delay_min'].fillna(dd['departure_delay_min']),
                })
                aligned = aligned.dropna(subset=['delay_minutes'])
                aligned['delay_minutes'] = aligned['delay_minutes'].clip(0, MAX_DELAY_CAP).astype(int)
                daily_dfs.append(aligned)
            except Exception as e:
                logger.warning(f"  Error loading {f}: {e}")
        if daily_dfs:
            df_new = pd.concat(daily_dfs, ignore_index=True)
            df = pd.concat([df, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=['train_no', 'date', 'station_code'], keep='last')
            logger.info(f"  Combined: {len(df):,} records")
    
    # Step 2: Feature engineering
    logger.info("STEP 2: Feature engineering...")
    df = add_time_features(df)
    df = add_metadata(df)
    df = engineer_features(df)
    logger.info(f"  Features computed: {len(df):,} records, {len(FEATURE_COLUMNS)} features")
    
    # Step 3: Split and train
    logger.info("STEP 3: Training...")
    dates = df['date'].sort_values().unique()
    n = len(dates)
    train_dates = dates[:int(n * 0.70)]
    val_dates = dates[int(n * 0.70):int(n * 0.85)]
    test_dates = dates[int(n * 0.85):]
    
    df_train = df[df['date'].isin(train_dates)]
    df_val = df[df['date'].isin(val_dates)]
    df_test = df[df['date'].isin(test_dates)]
    
    X_train = df_train[FEATURE_COLUMNS].fillna(0)
    X_val = df_val[FEATURE_COLUMNS].fillna(0)
    X_test = df_test[FEATURE_COLUMNS].fillna(0)
    y_train = df_train['delay_minutes']
    y_val = df_val['delay_minutes']
    y_test = df_test['delay_minutes']
    y_train_clf = df_train['is_on_time']
    y_val_clf = df_val['is_on_time']
    y_test_clf = df_test['is_on_time']
    
    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Train regressor
    regressor = CatBoostRegressor(
        iterations=1000, depth=10, learning_rate=0.03,
        l2_leaf_reg=2.0, subsample=0.85, random_strength=0.5,
        random_state=RANDOM_STATE, verbose=0, early_stopping_rounds=50,
    )
    regressor.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Train classifier
    classifier = CatBoostClassifier(
        iterations=1000, depth=9, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.85, random_strength=0.3,
        auto_class_weights='Balanced',
        random_state=RANDOM_STATE, verbose=0, early_stopping_rounds=50,
    )
    classifier.fit(X_train, y_train_clf, eval_set=(X_val, y_val_clf))
    
    # Evaluate
    y_pred = np.clip(regressor.predict(X_test), 0, 720)
    new_mae = mean_absolute_error(y_test, y_pred)
    new_acc = accuracy_score(y_test_clf, classifier.predict(X_test)) * 100
    logger.info(f"  New model: MAE={new_mae:.2f} | Accuracy={new_acc:.2f}%")
    
    # Step 4: Compare and deploy
    logger.info("STEP 4: Deploy decision...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODELS_DIR / 'final_model_metadata.json'
    current_mae = float('inf')
    if meta_path.exists():
        with open(meta_path) as f:
            current_mae = json.load(f).get('regression_mae', float('inf'))
    
    improvement = current_mae - new_mae
    deploy = improvement >= -IMPROVEMENT_THRESHOLD
    logger.info(f"  Current MAE: {current_mae} | New MAE: {new_mae:.2f} | Improvement: {improvement:.2f} | Deploy: {deploy}")
    
    if deploy:
        with open(MODELS_DIR / 'delay_regressor_final.pkl', 'wb') as f:
            pickle.dump(regressor, f)
        with open(MODELS_DIR / 'delay_classifier_final.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        
        metadata = {
            'winner': 'CatBoost',
            'regression_mae': round(new_mae, 2),
            'classification_accuracy': round(new_acc, 2),
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'deployed': True,
            'data_records': len(df),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Step 5: Generate lookup table
    if deploy:
        logger.info("STEP 5: Generating lookup table...")
        generate_lookup(regressor, df)
    
    # Step 6: Regenerate frontend JSON (fares + delay history + reliability)
    if deploy:
        logger.info("STEP 6: Regenerating frontend data...")
        try:
            from regenerate_frontend import regenerate_frontend_json
            regenerate_frontend_json()
        except Exception as e:
            logger.warning(f"  Frontend regeneration failed: {e}")
    
    logger.info(f"\nDONE in {time.time()-t_start:.1f}s | MAE: {new_mae:.2f} | Deployed: {deploy}")
    
    # Send notification
    try:
        import urllib.request
        duration = int(time.time() - t_start)
        msg = f"Model Retrainer {'SUCCESS' if deploy else 'SKIPPED'} | MAE: {new_mae:.2f} | Accuracy: {new_acc:.1f}% | Records: {len(df):,} | {duration}s"
        urllib.request.urlopen(
            urllib.request.Request("https://ntfy.sh/railmind-alerts-sbh", data=msg.encode())
        )
    except:
        pass


def generate_lookup(regressor, df):
    """Generate delay lookup table."""
    buckets = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 720)]
    bucket_labels = ['0-5', '5-15', '15-30', '30-60', '60-120', '120+']
    
    with open(TOP200_FILE) as f:
        top200 = [t['train_no'] for t in json.load(f)]
    
    # Get latest stats per train-station
    train_stations = df[df['train_no'].isin(top200)].groupby(['train_no', 'station_code']).agg(
        train_type_encoded=('train_type_encoded', 'first'),
        num_stops=('num_stops', 'first'),
        station_avg_delay=('station_avg_delay', 'last'),
        train_avg_delay=('train_avg_delay', 'last'),
        train_station_avg_delay=('train_station_avg_delay', 'last'),
        train_std_delay=('train_std_delay', 'last'),
        train_ontime_rate=('train_ontime_rate', 'last'),
        route_avg_delay=('route_avg_delay', 'last'),
        stop_sequence=('stop_sequence', 'median'),
        train_rolling_7d_avg=('train_rolling_7d_avg', 'last'),
        train_data_count=('train_data_count', 'last'),
    ).reset_index()
    
    rows = []
    for _, row in train_stations.iterrows():
        for bi, (low, high) in enumerate(buckets):
            mid = (low + high) / 2 if high < 720 else 150
            features = {
                'prev_station_delay': mid,
                'train_avg_delay': row.get('train_avg_delay', 30),
                'station_avg_delay': row.get('station_avg_delay', 30),
                'train_station_avg_delay': row.get('train_station_avg_delay', 30),
                'train_type_encoded': row.get('train_type_encoded', 6),
                'month': 6, 'is_fog_season': 0, 'stop_sequence': row.get('stop_sequence', 10),
                'num_stops': row.get('num_stops', 20), 'day_of_week': 3,
                'route_avg_delay': row.get('route_avg_delay', 30),
                'train_rolling_7d_avg': row.get('train_rolling_7d_avg', 30),
                'consecutive_late_days': 10,
                'train_std_delay': row.get('train_std_delay', 40),
                'train_ontime_rate': row.get('train_ontime_rate', 0.35),
                'is_festival': 0, 'is_weekend': 0, 'is_monsoon': 0, 'day_of_month': 15,
                'type_x_fog': 0, 'type_x_monsoon': 0, 'type_x_weekend': 0,
                'seq_x_train_avg': row.get('stop_sequence', 10) * row.get('train_avg_delay', 30),
                'train_data_count': row.get('train_data_count', 1000),
            }
            X = pd.DataFrame([features])[FEATURE_COLUMNS]
            pred = float(np.clip(regressor.predict(X)[0], 0, 720))
            rows.append({
                'train_no': row['train_no'],
                'station_code': row['station_code'],
                'prev_delay_bucket': bucket_labels[bi],
                'predicted_delay_min': round(pred, 1),
            })
    
    lookup_df = pd.DataFrame(rows)
    lookup_df.to_csv(MODELS_DIR / 'delay_lookup_table.csv', index=False)
    logger.info(f"  Lookup table: {len(lookup_df):,} rows")


if __name__ == '__main__':
    run_retrain()
