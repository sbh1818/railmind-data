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


def upsert_observations(rows, batch=300):
    url = f"{SUPABASE_URL}/rest/v1/delay_observations"
    ok = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        resp = requests.post(url, headers=HEADERS, data=json.dumps(chunk))
        if resp.status_code in (200, 201):
            ok += len(chunk)
        else:
            logger.warning(f"  upsert error {resp.status_code}: {resp.text[:150]}")
        time.sleep(0.2)
    return ok


def compute_30d_averages():
    """Recompute station_delay_30d from the last 30 days of delay_observations.
    Aggregates per (train_no, station_code): avg arrival delay + sample count."""
    logger.info("Computing 30-day station-wise averages...")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

    obs_url = f"{SUPABASE_URL}/rest/v1/delay_observations"
    all_obs = []
    offset = 0
    page = 1000
    while True:
        r = requests.get(
            f"{obs_url}?journey_date=gte.{cutoff}&select=train_no,station_code,arr_delay_min",
            headers={**HEADERS, "Range": f"{offset}-{offset+page-1}"},
        )
        if r.status_code not in (200, 206):
            logger.warning(f"  fetch obs error {r.status_code}: {r.text[:150]}")
            break
        rows = r.json()
        all_obs.extend(rows)
        if len(rows) < page:
            break
        offset += page

    logger.info(f"  loaded {len(all_obs)} observations (last 30d)")

    from collections import defaultdict
    agg = defaultdict(list)
    for o in all_obs:
        d = o.get("arr_delay_min")
        if d is not None:
            agg[(o["train_no"], o["station_code"])].append(d)

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

    upserted = upsert_observations(all_rows) if all_rows else 0
    logger.info(f"Upserted {upserted} observations")

    return ok, upserted


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
