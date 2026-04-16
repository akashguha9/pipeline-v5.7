"""
milk_test_polymarket_history.py
================================
STANDALONE script. No imports from your project.
No writes to your project files.
No changes to src/ or data/.

Purpose: test ONE hypothesis in isolation before touching fetch_polymarket.py.

HYPOTHESIS
----------
Polymarket GAMMA /prices-history endpoint returns daily price data
when called with condition_id (not clob token_id).

Endpoint under test:
  GET https://gamma-api.polymarket.com/prices-history
  ?market={condition_id}
  &startTs={unix_30_days_ago}
  &endTs={unix_now}
  &fidelity=1440

condition_id is read from your existing data/processed/polymarket_markets.csv.
The script picks the first valid row automatically.

Usage:
  python milk_test_polymarket_history.py
  python milk_test_polymarket_history.py --csv-path data/processed/polymarket_markets.csv
  python milk_test_polymarket_history.py --condition-id 0x<explicit_id>
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT    = 15


# ── helpers ───────────────────────────────────────────────────────────────────

def get_condition_id_from_csv(csv_path: str) -> tuple[str | None, str | None]:
    """
    Read the first condition_id and question from polymarket_markets.csv.
    Returns (condition_id, question). Does not import pandas — uses stdlib csv.
    """
    import csv
    p = Path(csv_path)
    if not p.exists():
        print(f"[FAIL] CSV not found: {p.resolve()}")
        return None, None

    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        print(f"[INFO] CSV headers: {headers}")

        # Find the condition_id column — field name varies
        cid_col = next(
            (h for h in headers if h.lower() in ("conditionid", "condition_id", "id")),
            None
        )
        q_col = next(
            (h for h in headers if h.lower() in ("question", "title", "slug")),
            None
        )

        if not cid_col:
            print(f"[FAIL] No condition_id column found. Headers: {headers}")
            return None, None

        for row in reader:
            cid = row.get(cid_col, "").strip()
            q   = row.get(q_col, "") if q_col else ""
            if cid and len(cid) > 10:          # basic sanity: must be a real hash
                return cid, q

    print(f"[FAIL] No valid condition_id row found in {p}")
    return None, None


def test_gamma_history(condition_id: str, question: str = "", days_back: int = 30) -> bool:
    """
    Hit the GAMMA /prices-history endpoint with condition_id.
    Prints full diagnosis regardless of outcome.
    Returns True on success.
    """
    now      = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())

    url    = f"{GAMMA_BASE}/prices-history"
    params = {
        "market":   condition_id,
        "startTs":  start_ts,
        "endTs":    now,
        "fidelity": 1440,    # daily candles = 1440 minutes/day
    }

    print("\n" + "="*60)
    print("HYPOTHESIS: GAMMA /prices-history works with condition_id")
    print("="*60)
    print(f"[TEST] condition_id : {condition_id}")
    print(f"[TEST] question     : {question[:60]}")
    print(f"[TEST] URL          : {url}")
    print(f"[TEST] params       : {json.dumps(params, indent=2)}")
    print(f"[TEST] full URL     : {url}?market={condition_id}&startTs={start_ts}&endTs={now}&fidelity=1440")
    print()

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[FAIL] Network error: {e}")
        return False

    print(f"[INFO] HTTP status  : {resp.status_code}")
    print(f"[INFO] Content-Type : {resp.headers.get('Content-Type', 'unknown')}")
    print(f"[INFO] Response body (first 400 chars):\n{resp.text[:400]}")
    print()

    if resp.status_code != 200:
        print(f"[FAIL] Non-200 status: {resp.status_code}")
        _diagnose_failure(resp.status_code, resp.text, condition_id)
        return False

    # Parse
    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON decode error: {e}")
        return False

    # Normalise: response may be {"history": [...]} or a bare list
    history = data if isinstance(data, list) else data.get("history", [])

    if not history:
        print("[FAIL] Endpoint returned 200 but history array is empty.")
        print("       This market may be too new or have no price history.")
        print("       → Try --days-back 90 or --days-back 365")
        print("       → Or try a different condition_id (older, resolved market)")
        return False

    print(f"[PASS] Got {len(history)} history rows.")
    print(f"[INFO] First row: {history[0]}")
    print(f"[INFO] Last row:  {history[-1]}")

    # Validate structure
    first = history[0]
    has_t = "t" in first or "timestamp" in first
    has_p = "p" in first or "price" in first or "close" in first
    if not (has_t and has_p):
        print(f"[WARN] Unexpected row structure: {first}")
        print("       Expected keys 't' (timestamp) and 'p' (price)")
        print("       Actual keys:", list(first.keys()))
    else:
        print("[INFO] Row structure looks correct (has 't' and 'p' keys)")

    print()
    print("="*60)
    print("MILK TEST: PASS")
    print("→ Hypothesis confirmed. Proceed to patch src/fetch_polymarket.py")
    print("="*60)
    return True


def _diagnose_failure(status: int, body: str, cid: str):
    print()
    print("="*60)
    print("MILK TEST: FAIL — Diagnosis")
    print("="*60)

    if status == 400:
        print("400 Bad Request. Most likely causes:")
        print("  A) condition_id format is wrong (not a 0x hex string)")
        print(f"     Your id: {cid[:20]}...")
        print("     Expected: 0x followed by 64 hex chars")
        if not cid.startswith("0x"):
            print("     ← LIKELY CAUSE: missing 0x prefix")
        print()
        print("  B) The 'market' param name is wrong for this endpoint version")
        print("     Alternatives to try: ?id= or ?conditionId=")
        print()
        print("  C) startTs / endTs format wrong")
        print("     Verify they are Unix seconds, not milliseconds")
        print()
        print("NEXT HYPOTHESIS: Try CLOB /timeseries with token_id")
        print("  Run: python milk_test_polymarket_history.py --clob-fallback")

    elif status == 404:
        print("404 Not Found. Most likely causes:")
        print("  A) Endpoint path changed")
        print("     Try: https://gamma-api.polymarket.com/markets/{condition_id}/prices-history")
        print("  B) Market does not exist on GAMMA")
        print("     Verify the condition_id is from GAMMA /markets, not CLOB")

    elif status == 429:
        print("429 Rate limited. Wait 60 seconds and retry.")

    elif status == 401 or status == 403:
        print("Auth required. The read endpoint should be public — check if you need an API key.")
    else:
        print(f"Unexpected {status}. Body: {body[:200]}")


def test_clob_fallback(condition_id: str, csv_path: str) -> bool:
    """
    Fallback hypothesis: use CLOB /timeseries with YES-side token_id.
    Reads token_id from the markets CSV.
    """
    import csv

    print("\n" + "="*60)
    print("FALLBACK HYPOTHESIS: CLOB /timeseries with token_id")
    print("="*60)

    # Extract token_id from CSV
    token_id = None
    p = Path(csv_path)
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            tok_col = next(
                (h for h in headers if "token" in h.lower() or "clob" in h.lower()),
                None
            )
            if tok_col:
                for row in reader:
                    t = row.get(tok_col, "").strip()
                    if t and len(t) > 10:
                        token_id = t
                        break

    if not token_id:
        print("[SKIP] Could not find token_id in CSV")
        print("       Provide manually: --token-id 0x<token>")
        return False

    now      = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())

    url    = "https://clob.polymarket.com/prices-history"
    params = {
        "market":   token_id,    # CLOB uses token_id here, not condition_id
        "startTs":  start_ts,
        "endTs":    now,
        "fidelity": 1440,
    }

    print(f"[TEST] token_id : {token_id[:20]}...")
    print(f"[TEST] URL      : {url}")
    print()

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"[FAIL] Network error: {e}")
        return False

    print(f"[INFO] HTTP status: {resp.status_code}")
    print(f"[INFO] Body (first 300 chars): {resp.text[:300]}")

    if resp.status_code == 200:
        data = resp.json()
        history = data if isinstance(data, list) else data.get("history", [])
        if history:
            print(f"\n[PASS] CLOB fallback works. Got {len(history)} rows.")
            print("→ Proceed with CLOB /timeseries using token_id")
            return True
    print("[FAIL] CLOB fallback also failed.")
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Milk test: Polymarket history endpoint")
    parser.add_argument(
        "--csv-path",
        default="data/processed/polymarket_markets.csv",
        help="Path to polymarket_markets.csv (default: data/processed/polymarket_markets.csv)",
    )
    parser.add_argument(
        "--condition-id",
        default=None,
        help="Explicit condition_id to test (skips CSV lookup)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="How many days of history to request (default: 30)",
    )
    parser.add_argument(
        "--clob-fallback",
        action="store_true",
        help="Also test CLOB /timeseries as fallback",
    )
    args = parser.parse_args()

    question = ""

    if args.condition_id:
        condition_id = args.condition_id
        print(f"[INFO] Using explicit condition_id: {condition_id}")
    else:
        condition_id, question = get_condition_id_from_csv(args.csv_path)
        if not condition_id:
            print("\n[ABORT] Cannot proceed without a valid condition_id.")
            print("  Either:")
            print("  1. Check that data/processed/polymarket_markets.csv exists and has rows")
            print("  2. Pass --condition-id 0x<your_id> explicitly")
            sys.exit(2)

    passed = test_gamma_history(condition_id, question, days_back=args.days_back)

    if not passed and args.clob_fallback:
        passed = test_clob_fallback(condition_id, args.csv_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
