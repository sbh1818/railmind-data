"""
Nightly Data Collector - Delay + CNF/WL
Runs every night at 12 AM IST via GitHub Actions.
Batch 1: Station-wise delay data (200 trains, page fetch)
Batch 2: CNF/WL availability data (50 trains × 4 classes, API call)
"""
import sys
import os
import json
import time
import csv
import re
import logging
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_DELAY = REPO_ROOT / 'data' / 'daily_collected' / 'delays'
DATA_DIR_CNF = REPO_ROOT / 'data' / 'daily_collected' / 'cnf_wl'
TOP200_FILE = REPO_ROOT / 'data' / 'top200_trains.json'
DELAY_BETWEEN_REQUESTS = 1
MAX_RETRIES = 2

# CNF/WL API config
CNF_API_URL = 'https://cttrainsapi.confirmtkt.com/api/v1/availability/2monthcalendar'
CNF_DEFAULT_PARAMS = {
    'querysource': 'ct-web',
    'enableTG': 'true',
    'tGPlan': 'CTG-4',
    'showTGPrediction': 'false',
    'showPredictionGlobal': 'true',
    'showTgBucketPrediction': 'false',
}
CNF_CLASSES = ['SL', '3A', '2A', '1A', 'CC', 'EC']


# ============================================================
# DELAY COLLECTION (Batch 1)
# ============================================================

def parse_delay_string(delay_str):
    if not delay_str or delay_str.strip() == '-':
        return None
    delay_str = delay_str.strip()
    if 'On Time' in delay_str or 'Right Time' in delay_str:
        return 0
    min_match = re.match(r'(\d+)\s*Min', delay_str, re.IGNORECASE)
    if min_match:
        return int(min_match.group(1))
    hr_match = re.match(r'(\d+):(\d+)\s*Hr', delay_str, re.IGNORECASE)
    if hr_match:
        return int(hr_match.group(1)) * 60 + int(hr_match.group(2))
    num_match = re.match(r'(\d+)', delay_str)
    if num_match:
        return int(num_match.group(1))
    return None


def fetch_delay_data(train_no, date_str, session):
    url = f'https://www.confirmtkt.com/train-running-status/{train_no}?date={date_str}'
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                return None
            start_marker = 'var data = {'
            idx = r.text.find(start_marker)
            if idx < 0:
                return None
            idx += len(start_marker) - 1
            depth = 0
            end_idx = idx
            for ci in range(idx, min(idx + 300000, len(r.text))):
                if r.text[ci] == '{':
                    depth += 1
                elif r.text[ci] == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = ci + 1
                        break
            data = json.loads(r.text[idx:end_idx])
            schedule = data.get('Schedule', [])
            if not schedule:
                return None
            stations = []
            for stn in schedule:
                arr_min = parse_delay_string(stn.get('arrivalDelay', '-'))
                dep_min = parse_delay_string(stn.get('departureDelay', '-'))
                stations.append({
                    'station_code': stn.get('StationCode', ''),
                    'station_name': stn.get('StationName', ''),
                    'scheduled_arrival': stn.get('ArrivalTime', ''),
                    'scheduled_departure': stn.get('DepartureTime', ''),
                    'arrival_delay_min': arr_min,
                    'departure_delay_min': dep_min,
                    'platform': stn.get('ExpectedPlatformNo', ''),
                    'distance_km': stn.get('Distance', ''),
                    'day': stn.get('Day', ''),
                    'has_data': arr_min is not None or dep_min is not None,
                })
            return stations
        except (json.JSONDecodeError, requests.RequestException) as e:
            logger.warning(f"  Delay error {train_no} (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def collect_delays(trains, session, date_param, date_iso):
    """Batch 1: Collect delay data for 200 trains."""
    DATA_DIR_DELAY.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR_DELAY / f'delays_{date_iso}.csv'
    
    fieldnames = [
        'collection_date', 'train_no', 'train_name', 'station_code',
        'station_name', 'stop_sequence', 'scheduled_arrival',
        'scheduled_departure', 'arrival_delay_min', 'departure_delay_min',
        'platform', 'distance_km', 'day',
    ]
    
    success = 0
    no_data = 0
    failed = 0
    total_stations = 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, train in enumerate(trains):
            train_no = train['train_no']
            stations = fetch_delay_data(train_no, date_param, session)
            
            if stations is None:
                failed += 1
            elif not any(s['has_data'] for s in stations):
                no_data += 1
            else:
                success += 1
                for seq, stn in enumerate(stations, 1):
                    if stn['has_data']:
                        writer.writerow({
                            'collection_date': date_iso,
                            'train_no': train_no,
                            'train_name': train['train_name'],
                            'station_code': stn['station_code'],
                            'station_name': stn['station_name'],
                            'stop_sequence': seq,
                            'scheduled_arrival': stn['scheduled_arrival'],
                            'scheduled_departure': stn['scheduled_departure'],
                            'arrival_delay_min': stn['arrival_delay_min'],
                            'departure_delay_min': stn['departure_delay_min'],
                            'platform': stn['platform'],
                            'distance_km': stn['distance_km'],
                            'day': stn['day'],
                        })
                        total_stations += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Delay: {i+1}/{len(trains)} | Success: {success}")
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    logger.info(f"  DELAY DONE | Success: {success} | No run: {no_data} | Failed: {failed} | Stations: {total_stations}")
    return success, failed


# ============================================================
# CNF/WL COLLECTION (Batch 2)
# ============================================================

def parse_wl_position(text):
    if not text:
        return None, None
    text = text.strip().upper()
    if 'WL' in text:
        nums = re.findall(r'\d+', text)
        return 'WL', int(nums[-1]) if nums else None
    elif 'RAC' in text:
        nums = re.findall(r'\d+', text)
        return 'RAC', int(nums[-1]) if nums else None
    elif 'AVAILABLE' in text or 'AVL' in text:
        nums = re.findall(r'\d+', text)
        return 'AVAILABLE', int(nums[0]) if nums else None
    elif 'REGRET' in text:
        return 'REGRET', 0
    return text, None


def fetch_cnf_data(train_no, from_stn, to_stn, travel_class, start_date):
    params = {
        'trainNumber': train_no,
        'sourceStationCode': from_stn,
        'destinationStationCode': to_stn,
        'trainClass': travel_class,
        'quota': 'GN',
        'startDate': start_date,
        **CNF_DEFAULT_PARAMS,
    }
    try:
        r = requests.get(CNF_API_URL, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return data.get('data', data)
    except Exception as e:
        logger.warning(f"  CNF error {train_no}/{travel_class}: {e}")
    return None


def collect_cnf_wl(trains, date_iso):
    """
    Batch 2: Collect CNF/WL availability for top 50 trains.
    Top 30 every night + rotate 20 from rest.
    """
    DATA_DIR_CNF.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR_CNF / f'cnf_wl_{date_iso}.csv'
    
    # Select trains: top 30 daily + 20 rotated from rest
    top_30 = trains[:30]
    rest = trains[30:200]
    day_of_month = datetime.now().day
    batch_idx = day_of_month % 9  # 9 batches to cover ~170 trains
    batch_start = batch_idx * 20
    rotated_20 = rest[batch_start:batch_start + 20]
    selected = top_30 + rotated_20
    
    start_date = datetime.now().strftime('%d-%m-%Y')
    snapshot_time = datetime.now().isoformat()
    
    fieldnames = [
        'snapshot_datetime', 'snapshot_date', 'train_no', 'train_name',
        'source_station', 'destination_station', 'travel_class', 'quota',
        'journey_date', 'days_before_journey', 'journey_day_of_week',
        'journey_month', 'journey_is_weekend',
        'availability_display', 'status_type', 'position_number',
        'prediction_text', 'prediction_percentage', 'confirm_status',
        'gradient', 'cache_time', 'batch_type',
    ]
    
    success = 0
    failed = 0
    total_rows = 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, train in enumerate(selected):
            train_no = train['train_no']
            from_stn = train['source_code']
            to_stn = train['destination_code']
            batch_type = 'top30' if i < 30 else 'rotated'
            
            for cls in CNF_CLASSES:
                data = fetch_cnf_data(train_no, from_stn, to_stn, cls, start_date)
                
                if data is None:
                    failed += 1
                else:
                    success += 1
                    for journey_date, avail_info in data.items():
                        if not isinstance(avail_info, dict):
                            continue
                        
                        avail_display = avail_info.get('availabilityDisplayName', '')
                        status_type, position_num = parse_wl_position(avail_display)
                        
                        # Compute days before journey
                        try:
                            parts = journey_date.split('-')
                            jd = datetime(int(parts[2]), int(parts[1]), int(parts[0]))
                            days_before = (jd - datetime.now()).days
                            j_dow = jd.weekday()
                            j_month = jd.month
                            j_weekend = j_dow >= 5
                        except:
                            days_before = None
                            j_dow = None
                            j_month = None
                            j_weekend = None
                        
                        writer.writerow({
                            'snapshot_datetime': snapshot_time,
                            'snapshot_date': date_iso,
                            'train_no': train_no,
                            'train_name': train['train_name'],
                            'source_station': from_stn,
                            'destination_station': to_stn,
                            'travel_class': cls,
                            'quota': 'GN',
                            'journey_date': journey_date,
                            'days_before_journey': days_before,
                            'journey_day_of_week': j_dow,
                            'journey_month': j_month,
                            'journey_is_weekend': j_weekend,
                            'availability_display': avail_display,
                            'status_type': status_type,
                            'position_number': position_num,
                            'prediction_text': avail_info.get('predictionDisplayName', ''),
                            'prediction_percentage': avail_info.get('predictionPercentage', ''),
                            'confirm_status': avail_info.get('confirmTktStatus', ''),
                            'gradient': avail_info.get('gradient', ''),
                            'cache_time': avail_info.get('cacheTime', ''),
                            'batch_type': batch_type,
                        })
                        total_rows += 1
                
                time.sleep(DELAY_BETWEEN_REQUESTS)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  CNF/WL: {i+1}/{len(selected)} trains | Calls: {success}")
    
    logger.info(f"  CNF/WL DONE | API calls: {success} | Failed: {failed} | Rows: {total_rows}")
    return success, failed


# ============================================================
# MAIN
# ============================================================

def run_collection():
    now = datetime.now()
    date_param = now.strftime('%d-%m-%Y')
    date_iso = now.strftime('%Y-%m-%d')
    
    trains = json.loads(open(TOP200_FILE, 'r', encoding='utf-8').read())
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    logger.info("=" * 60)
    logger.info(f"NIGHTLY DATA COLLECTOR | {date_param}")
    logger.info("=" * 60)
    
    # Batch 1: Delays
    logger.info("\n--- BATCH 1: DELAY DATA (200 trains) ---")
    delay_success, delay_failed = collect_delays(trains, session, date_param, date_iso)
    
    # Batch 2: CNF/WL
    logger.info("\n--- BATCH 2: CNF/WL DATA (50 trains × 4 classes) ---")
    cnf_success, cnf_failed = collect_cnf_wl(trains, date_iso)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info(f"  Delay: {delay_success} trains OK, {delay_failed} failed")
    logger.info(f"  CNF/WL: {cnf_success} API calls OK, {cnf_failed} failed")
    logger.info(f"  Total requests: {delay_success + delay_failed + cnf_success + cnf_failed}")
    logger.info("=" * 60)
    
    # Push daily avg delays to Supabase
    try:
        from supabase_sync import push_delays_to_supabase
        push_delays_to_supabase(date_iso)
        logger.info("  Supabase: daily delays synced")
    except Exception as e:
        logger.warning(f"  Supabase sync failed: {e}")
    
    # Send notification
    try:
        import urllib.request
        import os
        duration = int((datetime.now() - now).total_seconds() / 60)
        msg = f"Nightly Collector SUCCESS | Delay: {delay_success} OK, {delay_failed} failed | CNF/WL: {cnf_success} OK, {cnf_failed} failed | {duration} min"
        ntfy_topic = os.environ.get("NTFY_TOPIC", "")
        if ntfy_topic:
            urllib.request.urlopen(
                urllib.request.Request(f"https://ntfy.sh/{ntfy_topic}", data=msg.encode())
            )
    except:
        pass


if __name__ == '__main__':
    run_collection()
