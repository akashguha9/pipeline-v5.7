"""
fetch_polymarket.py  (src/ layer — fixed)
==========================================
Root cause of 400s: prior version used clob_token_ids against the GAMMA
price-history endpoint. The GAMMA API uses condition_id (or market slug),
NOT clob token_ids. CLOB token_ids are for the orderbook endpoint only.

CORRECT FLOW:
  1. GAMMA /markets → condition_id, slug, question, tokens[].token_id
  2. GAMMA /markets/{condition_id}/history → daily price series
  3. Output: polymarket_prices_daily.csv

No auth required for GAMMA read endpoints.
"""

from __future__ import annotations
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
GAMMA_BASE   = "https://gamma-api.polymarket.com"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MARKETS_CSV  = os.path.join(OUTPUT_DIR, "polymarket_markets.csv")
HISTORY_CSV  = os.path.join(OUTPUT_DIR, "polymarket_prices_daily.csv")

LOOKBACK_DAYS   = 180          # how far back to pull history
PAGE_LIMIT      = 100          # markets per page
REQUEST_DELAY   = 0.3          # seconds between calls (rate limit courtesy)
MAX_RETRIES     = 3
TIMEOUT         = 15

MACRO_KEYWORDS = [
    "fed", "federal reserve", "interest rate", "inflation", "cpi",
    "recession", "gdp", "unemployment", "election", "president",
    "war", "russia", "ukraine", "china", "taiwan", "oil", "opec",
    "bitcoin", "crypto", "default", "debt ceiling", "tariff",
    "iran", "israel", "nato", "g7", "imf", "world bank",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = MAX_RETRIES) -> Optional[dict | list]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** attempt
                log.warning("429 rate-limit, sleeping %ds", wait)
                time.sleep(wait)
                continue
            log.warning("HTTP %d for %s  body: %s", r.status_code, url, r.text[:120])
            return None
        except requests.RequestException as e:
            log.warning("Request error (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(attempt)
    return None


# ── Step 1: Fetch market metadata ─────────────────────────────────────────────

def fetch_markets(keyword_filter: bool = True) -> pd.DataFrame:
    """
    Pull GAMMA markets. Returns DataFrame with:
      condition_id | slug | question | end_date | volume | active | tokens_json
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_markets = []
    offset = 0

    while True:
        params = {
            "limit":  PAGE_LIMIT,
            "offset": offset,
            "active": "true",
            "closed": "false",
        }
        data = _get(f"{GAMMA_BASE}/markets", params=params)
        if not data:
            break

        # GAMMA returns a list directly
        batch = data if isinstance(data, list) else data.get("markets", [])
        if not batch:
            break

        all_markets.extend(batch)
        log.info("Fetched markets offset=%d  batch=%d  total=%d",
                 offset, len(batch), len(all_markets))

        if len(batch) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(REQUEST_DELAY)

    if not all_markets:
        log.error("No markets fetched from GAMMA")
        return pd.DataFrame()

    df = pd.json_normalize(all_markets)

    # Normalise key columns
    rename = {
        "conditionId": "condition_id",
        "id":          "condition_id",   # some responses use "id"
        "questionID":  "condition_id",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = ["condition_id"]
    for col in required:
        if col not in df.columns:
            log.error("Missing required column '%s'. Columns: %s", col, list(df.columns))
            return pd.DataFrame()

    if keyword_filter:
        q_col = next((c for c in ["question", "title", "slug"] if c in df.columns), None)
        if q_col:
            mask = df[q_col].str.lower().apply(
                lambda q: any(kw in str(q) for kw in MACRO_KEYWORDS)
            )
            filtered = df[mask].copy()
            log.info("Keyword filter: %d → %d markets", len(df), len(filtered))
            df = filtered if not filtered.empty else df.head(200)
        else:
            df = df.head(200)

    df.to_csv(MARKETS_CSV, index=False)
    log.info("Saved %d markets → %s", len(df), MARKETS_CSV)
    return df


# ── Step 2: Fetch price history via condition_id ───────────────────────────────

def fetch_market_history(condition_id: str, question: str = "") -> Optional[pd.DataFrame]:
    """
    GAMMA /prices-history endpoint.
    Returns DataFrame: date | close_prob
    
    CRITICAL FIX: endpoint uses condition_id, NOT clob token_id.
    The history endpoint signature:
      GET /prices-history?market={condition_id}&startTs={unix}&endTs={unix}&fidelity=1440
    
    fidelity=1440 = 1 day candles (minutes in a day)
    """
    end_ts   = int(datetime.utcnow().timestamp())
    start_ts = int((datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)).timestamp())

    url    = f"{GAMMA_BASE}/prices-history"
    params = {
        "market":   condition_id,
        "startTs":  start_ts,
        "endTs":    end_ts,
        "fidelity": 1440,      # daily
    }

    data = _get(url, params=params)
    if not data:
        # Fallback: try slug-based if we have it (some markets use slug in URL)
        return None

    # Response shape: {"history": [{"t": unix_ts, "p": float}, ...]}
    # or list of {t, p}
    history = data if isinstance(data, list) else data.get("history", [])
    if not history:
        log.debug("Empty history for %s (%s)", condition_id[:12], question[:40])
        return None

    rows = []
    for item in history:
        ts = item.get("t") or item.get("timestamp")
        p  = item.get("p") or item.get("price") or item.get("close")
        if ts and p is not None:
            rows.append({"date": datetime.utcfromtimestamp(ts).date(), "close_prob": float(p)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    # Resample to daily (fill forward gaps)
    df = df.set_index("date").resample("D").last().ffill().reset_index()
    df["condition_id"] = condition_id
    return df


# ── Step 3: Batch history pull ─────────────────────────────────────────────────

def fetch_all_history(markets_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Pull history for all markets in markets_df.
    Saves to polymarket_prices_daily.csv.
    Returns combined DataFrame.
    """
    if markets_df is None:
        if os.path.exists(MARKETS_CSV):
            markets_df = pd.read_csv(MARKETS_CSV)
        else:
            markets_df = fetch_markets()

    if markets_df.empty or "condition_id" not in markets_df.columns:
        log.error("No valid markets dataframe")
        return pd.DataFrame()

    all_history = []
    total = len(markets_df)
    success = 0

    q_col = next((c for c in ["question", "title", "slug"] if c in markets_df.columns), None)

    for idx, row in markets_df.iterrows():
        cid      = str(row["condition_id"])
        question = str(row[q_col]) if q_col else ""

        hist = fetch_market_history(cid, question)
        if hist is not None and not hist.empty:
            hist["question"] = question[:80]
            all_history.append(hist)
            success += 1
            if success % 10 == 0:
                log.info("History progress: %d/%d succeeded", success, idx + 1)

        time.sleep(REQUEST_DELAY)

    if not all_history:
        log.error("No history fetched for any market")
        return pd.DataFrame()

    combined = pd.concat(all_history, ignore_index=True)
    combined = combined.rename(columns={"date": "Date", "close_prob": "polymarket_prob"})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(HISTORY_CSV, index=False)
    log.info(
        "History complete: %d/%d markets, %d rows → %s",
        success, total, len(combined), HISTORY_CSV,
    )
    return combined


# ── Milk test ──────────────────────────────────────────────────────────────────

def milk_test() -> bool:
    """
    Verify one known live market returns history.
    Uses a well-known active election/macro market.
    Returns True if history rows > 0.
    """
    log.info("=== MILK TEST: single market history fetch ===")

    # Fetch first 5 markets, try to get history from any
    params = {"limit": 5, "active": "true", "closed": "false"}
    data   = _get(f"{GAMMA_BASE}/markets", params=params)
    if not data:
        log.error("MILK FAIL: cannot reach GAMMA /markets")
        return False

    markets = data if isinstance(data, list) else data.get("markets", [])
    if not markets:
        log.error("MILK FAIL: empty markets response")
        return False

    # Normalise first market
    m = markets[0]
    cid = m.get("conditionId") or m.get("id") or m.get("condition_id")
    if not cid:
        log.error("MILK FAIL: no condition_id found in %s", list(m.keys()))
        return False

    log.info("Testing history for condition_id=%s", cid[:16])
    hist = fetch_market_history(cid, m.get("question", "")[:60])

    if hist is None or hist.empty:
        log.error("MILK FAIL: no history rows returned")
        log.info("Diagnostic: check endpoint %s/prices-history?market=%s&fidelity=1440", GAMMA_BASE, cid)
        return False

    log.info("MILK PASS: got %d history rows for market %s", len(hist), cid[:16])
    log.info("Sample:\n%s", hist.tail(3).to_string())
    return True


if __name__ == "__main__":
    import sys

    if "--milk" in sys.argv:
        ok = milk_test()
        sys.exit(0 if ok else 1)

    if "--markets-only" in sys.argv:
        df = fetch_markets()
        print(f"Markets fetched: {len(df)}")
        sys.exit(0)

    # Full run
    log.info("Starting Polymarket ingestion pipeline")
    markets = fetch_markets(keyword_filter=True)
    if markets.empty:
        log.error("No markets — aborting")
        sys.exit(1)
    history = fetch_all_history(markets)
    if history.empty:
        log.error("No history rows produced")
        sys.exit(1)
    log.info("Done. %d rows in polymarket_prices_daily.csv", len(history))
