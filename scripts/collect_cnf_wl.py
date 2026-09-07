"""
CNF/WL Availability Collector (ConfirmTkt 2monthcalendar API)
-------------------------------------------------------------
Collects seat availability + confirmation prediction for CNF/WL-priority trains,
only for the classes each train actually offers. Writes datewise CSV (committed
to the public repo = permanent free dataset for ML).

Runs on GitHub Actions, split into parallel batches (round-robin).

Usage:
  python collect_cnf_wl.py --batch 1 --total 3
  python collect_cnf_wl.py --limit 10           (quick test)
  python collect_cnf_wl.py --aggregate --total 3  (merge batch CSVs + ntfy summary)

Env: NTFY_TOPIC (optional) for notifications.
"""
import os
import sys
import csv
import gzip
import json
import time
import random
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CNF_LIST = REPO_ROOT / "data" / "cnf_wl_trains.json"
OUT_DIR = REPO_ROOT / "data" / "daily_collected" / "cnf_wl"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()

CNF_API_URL = "https://cttrainsapi.confirmtkt.com/api/v1/availability/2monthcalendar"
CNF_DEFAULT_PARAMS = {
    "querysource": "ct-web",
    "enableTG": "true",
    "tGPlan": "CTG-4",
    "showTGPrediction": "false",
    "showPredictionGlobal": "true",
    "showTgBucketPrediction": "false",
}
# classes we support fetching; filtered to each train's actual classes
KNOWN_CLASSES = {"1A", "2A", "3A", "3E", "CC", "EC", "SL", "2S"}

PAUSE = 1.0
JITTER = 0.6
MAX_CONSEC_FAIL = 25

FIELDNAMES = [
    "snapshot_datetime", "train_no", "train_name",
    "source_station", "destination_station", "travel_class",
    "journey_date", "days_before_journey", "journey_day_of_week",
    "journey_month", "journey_is_weekend",
    "availability_display", "status_type", "position_number",
    "prediction_text", "prediction_percentage", "confirm_status",
    "batch",
]


def notify(title, msg, tags="steam_locomotive", priority="default"):
    if not NTFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Tags": tags, "Priority": priority}, timeout=10)
    except Exception:
        pass


def parse_wl_position(text):
    if not text:
        return None, None
    text = text.strip().upper()
    if "WL" in text:
        nums = re.findall(r"\d+", text)
        return "WL", int(nums[-1]) if nums else None
    if "RAC" in text:
        nums = re.findall(r"\d+", text)
        return "RAC", int(nums[-1]) if nums else None
    if "AVAILABLE" in text or "AVL" in text:
        nums = re.findall(r"\d+", text)
        return "AVAILABLE", int(nums[0]) if nums else None
    return "OTHER", None


def train_classes(t):
    raw = (t.get("classes") or "").replace(" ", "")
    cls = [c for c in raw.split(",") if c in KNOWN_CLASSES]
    return cls


def fetch_cnf(train_no, from_stn, to_stn, cls, start_date):
    params = {
        "trainNumber": train_no, "sourceStationCode": from_stn,
        "destinationStationCode": to_stn, "trainClass": cls,
        "quota": "GN", "startDate": start_date, **CNF_DEFAULT_PARAMS,
    }
    r = requests.get(CNF_API_URL, params=params, timeout=20)
    if r.status_code == 200:
        data = r.json()
        return data.get("data", data)
    return None


def collect(batch, total, limit=None):
    trains = json.loads(CNF_LIST.read_text(encoding="utf-8"))
    trains = [t for t in trains if train_classes(t)]  # only trains with known classes
    if limit:
        trains = trains[:limit]
    else:
        trains = [t for i, t in enumerate(trains) if i % total == (batch - 1)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_iso = datetime.now().strftime("%Y-%m-%d")
    suffix = "_test" if limit else f"_b{batch}"
    csv_path = OUT_DIR / f"cnf_wl_{date_iso}{suffix}.csv.gz"
    start_date = datetime.now().strftime("%d-%m-%Y")
    snap = datetime.now().isoformat()

    logger.info(f"Batch {batch}/{total}: {len(trains)} trains")
    ok = failed = rows = 0
    consec_fail = 0
    t0 = time.time()

    with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for i, t in enumerate(trains):
            tn = t["train_no"]
            for cls in train_classes(t):
                try:
                    data = fetch_cnf(tn, t["source_code"], t["dest_code"], cls, start_date)
                    if data is None:
                        failed += 1
                        consec_fail += 1
                        if consec_fail >= MAX_CONSEC_FAIL:
                            logger.error(f"Circuit breaker: {consec_fail} consecutive failures — stopping")
                            logger.info(f"Batch {batch} DONE (aborted) | ok={ok} failed={failed} rows={rows}")
                            return
                    else:
                        ok += 1
                        consec_fail = 0
                        for jdate, info in data.items():
                            if not isinstance(info, dict):
                                continue
                            disp = info.get("availabilityDisplayName", "")
                            st, pos = parse_wl_position(disp)
                            try:
                                p = jdate.split("-")
                                jd = datetime(int(p[2]), int(p[1]), int(p[0]))
                                dbj = (jd - datetime.now()).days
                                dow, mon, wknd = jd.weekday(), jd.month, jd.weekday() >= 5
                            except (ValueError, IndexError):
                                dbj = dow = mon = None
                                wknd = None
                            w.writerow({
                                "snapshot_datetime": snap,
                                "train_no": tn, "train_name": t["train_name"],
                                "source_station": t["source_code"], "destination_station": t["dest_code"],
                                "travel_class": cls, "journey_date": jdate,
                                "days_before_journey": dbj, "journey_day_of_week": dow,
                                "journey_month": mon, "journey_is_weekend": wknd,
                                "availability_display": disp, "status_type": st, "position_number": pos,
                                "prediction_text": info.get("predictionDisplayName", ""),
                                "prediction_percentage": info.get("predictionPercentage", ""),
                                "confirm_status": info.get("confirmTktStatus", ""),
                                "batch": f"{batch}/{total}",
                            })
                            rows += 1
                except Exception as e:
                    failed += 1
                    consec_fail += 1
                    if consec_fail >= MAX_CONSEC_FAIL:
                        logger.error(f"Circuit breaker tripped — stopping")
                        return
                time.sleep(PAUSE + random.uniform(0, JITTER))
            if (i + 1) % 50 == 0:
                el = int(time.time() - t0)
                logger.info(f"  {i+1}/{len(trains)} | ok={ok} failed={failed} rows={rows} | {el}s")

    logger.info(f"Batch {batch} DONE | ok={ok} failed={failed} rows={rows} -> {csv_path.name}")


def aggregate(total):
    """Merge today's batch CSVs into one, send ntfy summary."""
    date_iso = datetime.now().strftime("%Y-%m-%d")
    parts = sorted(OUT_DIR.glob(f"cnf_wl_{date_iso}_b*.csv.gz"))
    if not parts:
        notify("CNF/WL collection FAILED", f"No batch files for {date_iso}", "x", "high")
        logger.error("No batch files found")
        return
    merged = OUT_DIR / f"cnf_wl_{date_iso}.csv.gz"
    total_rows = 0
    trains_seen = set()
    with gzip.open(merged, "wt", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=FIELDNAMES)
        w.writeheader()
        for p in parts:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    w.writerow(row)
                    trains_seen.add(row["train_no"])
                    total_rows += 1
    # remove per-batch files after merge
    for p in parts:
        p.unlink()
    msg = (f"CNF/WL {date_iso}\n"
           f"Batches: {len(parts)}\n"
           f"Trains: {len(trains_seen)}\n"
           f"Rows: {total_rows:,}")
    notify("CNF/WL collection OK", msg, "white_check_mark")
    logger.info(msg.replace("\n", " | "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="test: first N trains, writes _test file")
    ap.add_argument("--aggregate", action="store_true", help="merge batch CSVs + ntfy summary")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.total)
    else:
        collect(args.batch, args.total, limit=args.limit)
