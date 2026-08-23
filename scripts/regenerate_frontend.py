"""
RailMind - Regenerate trains.json for frontend
Fetches latest data from ixigo API + merges with existing ConfirmTkt data + delay history.
Runs as part of biweekly retrainer → pushes to frontend repo → Vercel auto-deploys.
"""
import sys
import os
import json
import time
import glob
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIRMTKT_DATA = REPO_ROOT / 'data' / 'train_metadata.json'
TOP200_FILE = REPO_ROOT / 'data' / 'top200_trains.json'
HISTORICAL_CSV = REPO_ROOT / 'data' / 'historical' / 'daily_delays_flat.csv'
COLLECTED_DIR = REPO_ROOT / 'data' / 'daily_collected' / 'delays'
OUTPUT_PATH = REPO_ROOT / 'data' / 'trains_frontend.json'

IXIGO_API = 'https://www.ixigo.com/api/v2/trains/detailedInfo'
IXIGO_HEADERS = {'accept': '*/*', 'User-Agent': 'ConfirmTkt-Website/1.0'}
DELAY_BETWEEN_CALLS = 1


# ============================================================
# STEP 1: FETCH LATEST FROM IXIGO
# ============================================================

def fetch_ixigo_data(train_no):
    """Fetch latest train data from ixigo API."""
    try:
        r = requests.get(f'{IXIGO_API}/{train_no}', headers=IXIGO_HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json().get('data', {})
    except:
        pass
    return None


def refresh_from_ixigo(trains):
    """Update trains with latest ixigo data (fares, platforms, speed)."""
    logger.info("Refreshing data from ixigo API...")
    success = 0
    failed = 0

    for i, train in enumerate(trains):
        tn = train['train_no']
        data = fetch_ixigo_data(tn)

        if data:
            info = data.get('info', {})
            fare = data.get('fare', {})
            schedules = data.get('schedules', [])

            # Update fares
            if fare:
                train['fares'] = fare

            # Update speed/distance info
            if info.get('avgSpeedInKmph'):
                train['avg_speed_kmph'] = info['avgSpeedInKmph']
            if info.get('maxPermissibleSpeedInKmph'):
                train['max_speed_kmph'] = info['maxPermissibleSpeedInKmph']
            if info.get('totalDistanceInKm'):
                train['total_distance_km'] = info['totalDistanceInKm']
            if info.get('rakeType'):
                train['rake_type'] = info['rakeType']
            if info.get('averageRating'):
                train['ixigo_rating'] = float(info['averageRating'])

            # Update platform numbers from schedule
            if schedules:
                for sched in schedules:
                    station_code = sched.get('destination', {}).get('code', '')
                    platform = sched.get('platformAsString', '')
                    if station_code and platform:
                        # Update in train's schedule
                        for stop in train.get('schedule', []):
                            if stop.get('station_code') == station_code:
                                stop['platform_no'] = platform
                                break

            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  ixigo refresh: {i+1}/{len(trains)} | Success: {success}")

        time.sleep(DELAY_BETWEEN_CALLS)

    logger.info(f"  ixigo refresh done: {success} OK, {failed} failed")
    return trains


# ============================================================
# STEP 2: COMPUTE DELAY HISTORY (last 7/14 days)
# ============================================================

def compute_delay_history(trains):
    """Compute per-day delay bars for last 7 and 14 days."""
    logger.info("Computing delay history...")

    # Load historical + collected data
    dfs = []

    if HISTORICAL_CSV.exists():
        df_hist = pd.read_csv(HISTORICAL_CSV, dtype={'train_no': str, 'delay_minutes': str}, low_memory=False)
        df_hist['delay_minutes'] = pd.to_numeric(df_hist['delay_minutes'], errors='coerce')
        df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
        dfs.append(df_hist[['train_no', 'date', 'delay_minutes']].dropna())

    # Load daily collected
    daily_files = sorted(glob.glob(str(COLLECTED_DIR / 'delays_*.csv')))
    for f in daily_files:
        try:
            dd = pd.read_csv(f, dtype={'train_no': str})
            if 'arrival_delay_min' in dd.columns:
                dd_clean = pd.DataFrame({
                    'train_no': dd['train_no'].astype(str),
                    'date': pd.to_datetime(dd['collection_date']),
                    'delay_minutes': dd['arrival_delay_min'].fillna(dd['departure_delay_min']),
                })
                dfs.append(dd_clean.dropna())
        except:
            pass

    if not dfs:
        logger.warning("  No delay data found")
        return trains

    df = pd.concat(dfs, ignore_index=True)
    df['delay_minutes'] = df['delay_minutes'].clip(0, 720).astype(int)
    max_date = df['date'].max()

    for train in trains:
        tn = train['train_no']
        tdf = df[df['train_no'] == tn]

        if len(tdf) == 0:
            train['delay_history'] = {'7d': [], '14d': []}
            continue

        daily = tdf.groupby(tdf['date'].dt.date)['delay_minutes'].mean().reset_index()
        daily.columns = ['date', 'avg_delay']
        daily = daily.sort_values('date', ascending=False)

        history = {}
        for label, days in [('7d', 7), ('14d', 14)]:
            cutoff = (max_date - pd.Timedelta(days=days)).date()
            subset = daily[daily['date'] >= cutoff].head(days)
            runs = [{'date': row['date'].strftime('%d %b'), 'delay': round(row['avg_delay'], 0)}
                    for _, row in subset.iterrows()]
            history[label] = list(reversed(runs))

        train['delay_history'] = history

    logger.info(f"  Delay history computed for {len(trains)} trains")
    return trains


# ============================================================
# STEP 3: COMPUTE RELIABILITY STATS
# ============================================================

def compute_reliability(trains):
    """Compute on-time %, avg delay, best/worst day per train."""
    logger.info("Computing reliability stats...")

    dfs = []
    if HISTORICAL_CSV.exists():
        df = pd.read_csv(HISTORICAL_CSV, dtype={'train_no': str, 'delay_minutes': str}, low_memory=False)
        df['delay_minutes'] = pd.to_numeric(df['delay_minutes'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['delay_minutes', 'date'])
        df['delay_minutes'] = df['delay_minutes'].clip(0, 720).astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek
        dfs.append(df)

    if not dfs:
        return trains

    df = pd.concat(dfs)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    stats = df.groupby('train_no').agg(
        avg_delay=('delay_minutes', 'mean'),
        on_time_pct=('delay_minutes', lambda x: (x <= 10).mean() * 100),
        total_runs=('date', 'nunique'),
    ).reset_index()

    day_stats = df.groupby(['train_no', 'day_of_week'])['delay_minutes'].apply(
        lambda x: (x <= 10).mean() * 100).reset_index()
    day_stats.columns = ['train_no', 'day_of_week', 'ontime_pct']
    best_day = day_stats.loc[day_stats.groupby('train_no')['ontime_pct'].idxmax()]
    worst_day = day_stats.loc[day_stats.groupby('train_no')['ontime_pct'].idxmin()]

    for train in trains:
        tn = train['train_no']
        s = stats[stats['train_no'] == tn]
        if len(s) > 0:
            row = s.iloc[0]
            bd = best_day[best_day['train_no'] == tn]
            wd = worst_day[worst_day['train_no'] == tn]
            train['reliability'] = {
                'avg_delay': round(row['avg_delay'], 1),
                'on_time_pct': round(row['on_time_pct'], 1),
                'total_runs': int(row['total_runs']),
                'best_day': day_names[int(bd.iloc[0]['day_of_week'])] if len(bd) > 0 else '',
                'best_day_pct': round(bd.iloc[0]['ontime_pct'], 1) if len(bd) > 0 else 0,
                'worst_day': day_names[int(wd.iloc[0]['day_of_week'])] if len(wd) > 0 else '',
                'worst_day_pct': round(wd.iloc[0]['ontime_pct'], 1) if len(wd) > 0 else 0,
            }

    logger.info("  Reliability stats computed")
    return trains


# ============================================================
# MAIN
# ============================================================

def regenerate_frontend_json():
    logger.info("=" * 60)
    logger.info("REGENERATING trains.json FOR FRONTEND")
    logger.info("=" * 60)

    # Load base train data (from ConfirmTkt metadata)
    with open(CONFIRMTKT_DATA, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Load existing trains data structure from top200
    with open(TOP200_FILE, 'r') as f:
        top200_list = json.load(f)
    top200_nos = [t['train_no'] for t in top200_list]

    # Build base trains list from metadata
    trains = []
    for t in meta:
        trains.append({
            'train_no': t.get('train_no', ''),
            'train_name': t.get('train_name', ''),
            'source': t.get('source', t.get('train_name', '')),
            'source_code': t.get('source_code', ''),
            'destination': t.get('destination', ''),
            'destination_code': t.get('destination_code', ''),
            'train_type': t.get('train_type', 'OTHER'),
            'duration': t.get('duration', ''),
            'classes': t.get('classes', []),
            'days_of_run': t.get('days_of_run', {}),
            'has_pantry': t.get('has_pantry', False),
            'rating': t.get('rating', 0),
            'food_rating': t.get('food_rating', 0),
            'punctuality_rating': t.get('punctuality_rating', 0),
            'cleanliness_rating': t.get('cleanliness_rating', 0),
            'rating_count': t.get('rating_count', 0),
            'schedule': t.get('schedule', []),
            'fares': {},
            'reliability': {},
            'delay_history': {'7d': [], '14d': []},
        })

    # Step 1: Refresh from ixigo
    trains = refresh_from_ixigo(trains)

    # Step 2: Compute delay history
    trains = compute_delay_history(trains)

    # Step 3: Compute reliability
    trains = compute_reliability(trains)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(trains, f)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    logger.info(f"  Saved: {OUTPUT_PATH} ({size_kb:.0f} KB, {len(trains)} trains)")

    # Send notification
    try:
        import urllib.request
        msg = f"Frontend JSON regenerated | {len(trains)} trains | {size_kb:.0f} KB | Fares + delays + reliability updated"
        urllib.request.urlopen(
            urllib.request.Request("https://ntfy.sh/railmind-alerts-sbh", data=msg.encode())
        )
    except:
        pass

    return OUTPUT_PATH


if __name__ == '__main__':
    regenerate_frontend_json()
