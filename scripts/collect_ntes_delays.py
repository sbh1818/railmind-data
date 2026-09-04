"""
NTES Delay Collector
--------------------
For each train in the delay set, fetch YESTERDAY's completed run from NTES
(live_status), extract real per-station arrival/departure delays, and store
them in Supabase `delay_observations`. Then recompute `station_delay_30d`.

Runs every 3rd day via GitHub Actions. NTES keeps a ~6-day window, so a 3-day
cadence never loses data.

Guardrails (be a good guest to a public govt service):
  - polite pause with random jitter between calls
  - exponential backoff on network trouble
  - circuit breaker: stop if too many consecutive failures
  - ramp-up: --limit flag to test on a subset first

Env:
  SUPABASE_URL, SUPABASE_KEY (secret), NTFY_TOPIC (optional)
"""
import os
import sys
import json
import time
import random
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from ntes import NTESClient
from ntes.exceptions import NTESError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_KEY required")
    sys.exit(1)

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}

# delay train list (built from the priority analysis; one train_no per line)
DELAY_LIST = Path(__file__).resolve().parent.parent / "data" / "delay_trains.json"
# raw per-run delay CSVs (permanent ML dataset, committed to git)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "daily_collected" / "ntes_delays"

PAUSE = 1.5              # base seconds between calls
JITTER = 1.0            # up to +1s random
MAX_CONSEC_FAIL = 15    # circuit breaker


def parse_delay(s):
    """NTES delay strings: 'On Time' -> 0; 'HH:MM' -> minutes; '' -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s in ("-", "Source", "Destination"):
        return None
    if "On Time" in s or "Right Time" in s:
        return 0
    if ":" in s:
        try:
            h, m = s.split(":")[:2]
            return int(h) * 60 + int(m)
        except ValueError:
            return None
    if s.isdigit():
        return int(s)
    return None


def fetch_run(client, train_no, date_str):
    """Return list of {station_code, arr_delay, dep_delay} for a completed run,
    or None if no data / not run that day."""
    r = client.live_status(train_no, date_str)
    if not isinstance(r, dict):
        return None
    stns = r.get("STNS", [])
    if not stns:
        return None
    out = []
    for s in stns:
        code = (s.get("SC") or "").strip()
        if not code:
            continue
        out.append({
            "station_code": code,
            "arr_delay_min": parse_delay(s.get("DARR")),
            "dep_delay_min": parse_delay(s.get("DDEP")),
        })
    # keep only stations that actually have a delay reading
    out = [o for o in out if o["arr_delay_min"] is not None or o["dep_delay_min"] is not None]
    return out or None


def compute_30d_averages():
    """Recompute station_delay_30d from the last 30 days of CSV files."""
    logger.info("Computing 30-day station-wise averages from CSVs...")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date()

    from collections import defaultdict
    import csv as csvmod
    agg = defaultdict(list)

    files = sorted(RAW_DIR.glob("delays_*.csv"))
    used = 0
    for fp in files:
        # filename: delays_YYYY-MM-DD.csv
        try:
            fdate = datetime.strptime(fp.stem.replace("delays_", ""), "%Y-%m-%d").date()
        except ValueError:
            continue
        if fdate < cutoff:
            continue
        used += 1
        with open(fp, encoding="utf-8") as f:
            for row in csvmod.DictReader(f):
                d = row.get("arr_delay_min")
                if d not in (None, "", "None"):
                    try:
                        agg[(row["train_no"], row["station_code"])].append(int(float(d)))
                    except ValueError:
                        pass

    logger.info(f"  aggregated {used} daily files")

    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_rows = [{
        "train_no": tn,
        "station_code": sc,
        "avg_arr_delay": round(sum(delays) / len(delays), 1),
        "sample_count": len(delays),
        "updated_at": now_ts,
    } for (tn, sc), delays in agg.items()]

    url = f"{SUPABASE_URL}/rest/v1/station_delay_30d"
    ok = 0
    for i in range(0, len(summary_rows), 300):
        chunk = summary_rows[i:i + 300]
        resp = requests.post(url, headers=HEADERS, data=json.dumps(chunk))
        if resp.status_code in (200, 201):
            ok += len(chunk)
        else:
            logger.warning(f"  summary upsert error {resp.status_code}: {resp.text[:150]}")
        time.sleep(0.2)
    logger.info(f"  station_delay_30d updated: {ok} rows")
    return ok


def collect(limit=None, batch=None, total=None):
    trains = json.loads(DELAY_LIST.read_text(encoding="utf-8"))
    if limit:
        trains = trains[:limit]
    if batch is not None and total:
        trains = trains[(batch - 1)::total]  # round-robin split
        logger.info(f"Batch {batch}/{total}: {len(trains)} trains")

    client = NTESClient()
    import functools
    client.session.request = functools.partial(client.session.request, timeout=(10, 30))

    # yesterday in IST
    ist = timezone(timedelta(hours=5, minutes=30))
    yday = (datetime.now(ist) - timedelta(days=1))
    date_str = yday.strftime("%d-%b-%Y")
    journey_date = yday.strftime("%Y-%m-%d")

    logger.info(f"Collecting delays for {len(trains)} trains | run date {date_str}")

    all_rows = []
    ok = no_data = failed = 0
    consec_fail = 0

    for i, tn in enumerate(trains):
        tn = str(tn)
        try:
            stations = None
            for attempt in range(3):
                try:
                    stations = fetch_run(client, tn, date_str)
                    break
                except NTESError:
                    stations = None  # semantic 'no' — don't retry
                    break
                except Exception:
                    wait = [3, 10, 0][attempt]
                    if attempt < 2:
                        time.sleep(wait)
                    else:
                        raise
            if stations is None:
                no_data += 1
            else:
                ok += 1
                for s in stations:
                    all_rows.append({
                        "train_no": tn,
                        "station_code": s["station_code"],
                        "journey_date": journey_date,
                        "arr_delay_min": s["arr_delay_min"],
                        "dep_delay_min": s["dep_delay_min"],
                    })
            consec_fail = 0
        except Exception as e:
            failed += 1
            consec_fail += 1
            logger.warning(f"  {tn} failed: {type(e).__name__}")
            if consec_fail >= MAX_CONSEC_FAIL:
                logger.error(f"Circuit breaker: {consec_fail} consecutive failures — stopping")
                break

        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(trains)} | ok={ok} no_data={no_data} failed={failed} rows={len(all_rows)}")
        time.sleep(PAUSE + random.uniform(0, JITTER))

    logger.info(f"Fetched: ok={ok} no_data={no_data} failed={failed} | rows={len(all_rows)}")

    # Write raw observations to a per-batch CSV (committed to git by the workflow).
    written = 0
    if all_rows:
        import csv as csvmod
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_b{batch}" if batch else ""
        csv_path = RAW_DIR / f"delays_{journey_date}{suffix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=["train_no", "station_code", "journey_date",
                                                 "arr_delay_min", "dep_delay_min"])
            w.writeheader()
            for r_ in all_rows:
                w.writerow(r_)
            written = len(all_rows)
        logger.info(f"Wrote {written} rows to {csv_path.name}")

    return ok, written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N trains (testing)")
    ap.add_argument("--batch", type=int, default=None, help="1-indexed batch number")
    ap.add_argument("--total", type=int, default=None, help="total number of batches")
    ap.add_argument("--aggregate", action="store_true", help="compute 30-day averages + send summary")
    args = ap.parse_args()

    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime.now(ist)

    if args.aggregate:
        # Aggregation stage: recompute 30d averages, send final summary ntfy
        rows = compute_30d_averages()
        end = datetime.now(ist)
        dur = int((end - start).total_seconds() / 60)
        if NTFY_TOPIC:
            try:
                import urllib.request
                msg = (f"Delay Collector DONE\n"
                       f"Start: {start.strftime('%d-%b %H:%M')} IST\n"
                       f"End: {end.strftime('%d-%b %H:%M')} IST\n"
                       f"Duration: {dur} min\n"
                       f"30-day averages updated: {rows} station rows")
                urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode()))
            except Exception:
                pass
    else:
        collect(limit=args.limit, batch=args.batch, total=args.total)
