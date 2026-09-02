"""
Push daily average delays to Supabase + cleanup old rows (keep only 14 days).
"""
import csv
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECTED_DIR = REPO_ROOT / 'data' / 'daily_collected' / 'delays'

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}


def push_delays_to_supabase(date_iso: str):
    """Read today's delay CSV, compute avg per train, push to Supabase."""
    
    csv_path = COLLECTED_DIR / f'delays_{date_iso}.csv'
    if not csv_path.exists():
        logger.warning(f"  No delay file for {date_iso}")
        return
    
    # Compute avg delay per train
    train_delays = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tn = row.get('train_no', '')
            delay = row.get('arrival_delay_min') or row.get('departure_delay_min')
            if tn and delay:
                try:
                    d = float(delay)
                    if tn not in train_delays:
                        train_delays[tn] = []
                    train_delays[tn].append(d)
                except:
                    pass
    
    if not train_delays:
        logger.warning("  No delay data to push")
        return
    
    # Build rows for upsert
    rows = []
    for tn, delays in train_delays.items():
        avg = round(sum(delays) / len(delays), 1)
        rows.append({
            "train_no": tn,
            "collection_date": date_iso,
            "avg_delay": avg,
        })
    
    # Upsert to Supabase (batch)
    url = f"{SUPABASE_URL}/rest/v1/daily_delays"
    
    # Push in batches of 100
    for i in range(0, len(rows), 100):
        batch = rows[i:i+100]
        r = requests.post(url, json=batch, headers=HEADERS)
        if r.status_code not in (200, 201):
            logger.warning(f"  Supabase insert error: {r.status_code} {r.text[:200]}")
    
    logger.info(f"  Supabase: pushed {len(rows)} train delays for {date_iso}")
    
    # Cleanup: delete rows older than 14 days
    cutoff = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    delete_url = f"{SUPABASE_URL}/rest/v1/daily_delays?collection_date=lt.{cutoff}"
    r = requests.delete(delete_url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    if r.status_code in (200, 204):
        logger.info(f"  Supabase: cleaned rows older than {cutoff}")
    else:
        logger.warning(f"  Supabase cleanup error: {r.status_code}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test with today's date
    today = datetime.now().strftime('%Y-%m-%d')
    push_delays_to_supabase(today)
