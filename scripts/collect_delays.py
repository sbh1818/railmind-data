"""
RailMind - Nightly Delay Data Collector (GitHub Actions version)
Fetches station-wise delay data for top 200 trains from ConfirmTkt.
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
# CONFIGURATION (relative to repo root)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data' / 'daily_collected'
TOP200_FILE = REPO_ROOT / 'data' / 'top200_trains.json'
DELAY_BETWEEN_REQUESTS = 2
MAX_RETRIES = 2


def load_trains():
    with open(TOP200_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_collection_date():
    now = datetime.now()
    return now.strftime('%d-%m-%Y'), now.strftime('%Y-%m-%d')


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
            
            # Extract embedded JSON via brace counting
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
            logger.warning(f"  Error for {train_no} (attempt {attempt+1}): {e}")
            time.sleep(2)
    
    return None


def run_collection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    date_param, date_iso = get_collection_date()
    csv_path = DATA_DIR / f'delays_{date_iso}.csv'
    trains = load_trains()
    
    logger.info("=" * 60)
    logger.info(f"NIGHTLY DELAY COLLECTOR | Date: {date_param} | Trains: {len(trains)}")
    logger.info("=" * 60)
    
    fieldnames = [
        'collection_date', 'train_no', 'train_name', 'station_code',
        'station_name', 'stop_sequence', 'scheduled_arrival',
        'scheduled_departure', 'arrival_delay_min', 'departure_delay_min',
        'platform', 'distance_km', 'day',
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    success_trains = 0
    no_data_trains = 0
    failed_trains = 0
    total_stations = 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, train in enumerate(trains):
            train_no = train['train_no']
            train_name = train['train_name']
            
            stations = fetch_delay_data(train_no, date_param, session)
            
            if stations is None:
                failed_trains += 1
            elif not any(s['has_data'] for s in stations):
                no_data_trains += 1
            else:
                success_trains += 1
                for seq, stn in enumerate(stations, 1):
                    if stn['has_data']:
                        writer.writerow({
                            'collection_date': date_iso,
                            'train_no': train_no,
                            'train_name': train_name,
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
                logger.info(f"  Progress: {i+1}/{len(trains)} | Success: {success_trains}")
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    logger.info("=" * 60)
    logger.info(f"COMPLETE | Success: {success_trains} | No run: {no_data_trains} | Failed: {failed_trains} | Stations: {total_stations}")
    logger.info("=" * 60)


if __name__ == '__main__':
    run_collection()
