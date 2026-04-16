"""
build_dataset.py  (root layer — poly-only mode)
================================================
Builds final_dataset.csv from:
  1. market_prices.csv        (existing, working)
  2. polymarket_prices_daily.csv  (from fixed fetch_polymarket.py)
  3. Kalshi skipped if empty (no crash, just warning)

Output schema:
  Date | Ticker | Close | polymarket_prob

Integration contract with signal_engine.py:
  - Adds polymarket_prob column to the timeseries dataset
  - signal_engine reads output_timeseries_dataset.csv
  - This script produces final_dataset.csv which feeds build_timeseries_dataset.py
"""

from __future__ import annotations
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"

MARKET_PRICES_CSV    = DATA_DIR / "market_prices.csv"
POLY_HISTORY_CSV     = DATA_DIR / "polymarket_prices_daily.csv"
KALSHI_HISTORY_CSV   = DATA_DIR / "kalshi_prices_daily.csv"
FINAL_DATASET_CSV    = DATA_DIR / "final_dataset.csv"

# Match window: days to look back/forward when joining poly prob to ticker date
JOIN_TOLERANCE_DAYS = 3


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_market_prices() -> pd.DataFrame:
    if not MARKET_PRICES_CSV.exists():
        raise FileNotFoundError(f"market_prices.csv not found at {MARKET_PRICES_CSV}")

    df = pd.read_csv(MARKET_PRICES_CSV)
    df.columns = df.columns.str.strip()

    # Normalise date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError(f"No date column in market_prices. Columns: {list(df.columns)}")
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Normalise close / ticker
    ticker_col = next((c for c in df.columns if c.lower() in ["ticker", "symbol"]), None)
    close_col  = next((c for c in df.columns if c.lower() in ["close", "adj close", "price"]), None)
    if ticker_col:
        df = df.rename(columns={ticker_col: "Ticker"})
    if close_col:
        df = df.rename(columns={close_col: "Close"})

    log.info("Market prices: %d rows, tickers: %s", len(df),
             sorted(df["Ticker"].unique())[:5] if "Ticker" in df.columns else "?")
    return df


def load_polymarket_history() -> pd.DataFrame:
    """
    Loads polymarket_prices_daily.csv.
    Returns empty DataFrame (not error) if file missing — caller handles gracefully.
    Schema expected: Date | condition_id | polymarket_prob | question (optional)
    """
    if not POLY_HISTORY_CSV.exists():
        log.warning("polymarket_prices_daily.csv not found — running without belief layer")
        return pd.DataFrame()

    df = pd.read_csv(POLY_HISTORY_CSV)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if "polymarket_prob" not in df.columns:
        # Try close_prob as fallback
        if "close_prob" in df.columns:
            df = df.rename(columns={"close_prob": "polymarket_prob"})
        else:
            log.warning("No polymarket_prob column in history CSV")
            return pd.DataFrame()

    df["polymarket_prob"] = pd.to_numeric(df["polymarket_prob"], errors="coerce")
    log.info("Polymarket history: %d rows, %d unique markets",
             len(df), df["condition_id"].nunique() if "condition_id" in df.columns else 0)
    return df


def load_kalshi_history() -> pd.DataFrame:
    """
    Kalshi assumed unusable per brief. Returns empty DataFrame, logs warning only.
    """
    if not KALSHI_HISTORY_CSV.exists() or KALSHI_HISTORY_CSV.stat().st_size < 50:
        log.warning("Kalshi history empty or missing — skipping (expected per constraints)")
        return pd.DataFrame()

    df = pd.read_csv(KALSHI_HISTORY_CSV)
    if df.empty:
        log.warning("Kalshi history CSV is empty — skipping")
        return pd.DataFrame()

    log.info("Kalshi history loaded: %d rows", len(df))
    return df


# ── Join strategy ─────────────────────────────────────────────────────────────

def derive_ticker_keywords(tickers: list[str]) -> dict[str, list[str]]:
    """
    Maps ticker symbols to keyword lists for fuzzy join with Polymarket questions.
    E.g. SPY → ["s&p", "spy", "500", "us stock", "equity"]
    """
    TICKER_KEYWORDS = {
        "SPY":  ["s&p", "spy", "500", "us stock", "equity", "recession"],
        "QQQ":  ["nasdaq", "qqq", "tech", "technology"],
        "GLD":  ["gold", "gld", "precious metals"],
        "USO":  ["oil", "uso", "crude", "opec", "energy"],
        "TLT":  ["treasury", "tlt", "bond", "interest rate", "fed", "federal reserve"],
        "EEM":  ["emerging market", "eem", "developing"],
        "EWZ":  ["brazil", "ewz", "brl"],
        "UNG":  ["natural gas", "ung", "lng"],
        "BTC":  ["bitcoin", "btc", "crypto"],
        "VXX":  ["volatility", "vxx", "vix", "fear"],
        "DXY":  ["dollar", "dxy", "usd", "currency"],
        "FCG":  ["natural gas", "fcg", "energy"],
        "MSFT": ["microsoft", "msft", "tech"],
        "AAPL": ["apple", "aapl"],
        "JPM":  ["jpmorgan", "jpm", "bank", "financial"],
    }
    result = {}
    for t in tickers:
        t_upper = t.upper()
        result[t_upper] = TICKER_KEYWORDS.get(t_upper, [t.lower()])
    return result


def match_poly_to_ticker(
    poly_df: pd.DataFrame,
    ticker: str,
    keywords: list[str],
) -> pd.DataFrame:
    """
    Filter poly_df rows whose question matches any keyword for this ticker.
    Returns subset with Date, polymarket_prob columns.
    """
    if poly_df.empty or "polymarket_prob" not in poly_df.columns:
        return pd.DataFrame()

    q_col = next((c for c in poly_df.columns if c in ["question", "title", "slug"]), None)
    if q_col is None:
        # No question column — use ALL markets, take daily mean
        agg = poly_df.groupby("Date")["polymarket_prob"].mean().reset_index()
        agg.columns = ["Date", "polymarket_prob"]
        return agg

    mask = poly_df[q_col].str.lower().apply(
        lambda q: any(kw in str(q).lower() for kw in keywords)
    )
    matched = poly_df[mask].copy()

    if matched.empty:
        # Fallback: use aggregate mean across all markets for this date
        agg = poly_df.groupby("Date")["polymarket_prob"].mean().reset_index()
        agg.columns = ["Date", "polymarket_prob"]
        log.debug("Ticker %s: no keyword match, using market-wide mean", ticker)
        return agg

    # Aggregate multiple matching markets per day: take mean
    agg = matched.groupby("Date")["polymarket_prob"].mean().reset_index()
    log.debug("Ticker %s: %d matching market rows, %d daily obs",
              ticker, len(matched), len(agg))
    return agg


def join_belief_to_prices(
    prices_df: pd.DataFrame,
    poly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join poly belief onto market prices.
    Tolerant merge: if exact date miss, use nearest within JOIN_TOLERANCE_DAYS.
    Missing belief → NaN (not dropped — downstream data_void_engine handles it).
    """
    if poly_df.empty:
        prices_df["polymarket_prob"] = np.nan
        log.warning("No Polymarket history — polymarket_prob set to NaN (data_void_engine will handle)")
        return prices_df

    tickers = prices_df["Ticker"].unique() if "Ticker" in prices_df.columns else ["ALL"]
    kw_map  = derive_ticker_keywords(list(tickers))

    chunks = []
    for ticker in tickers:
        sub    = prices_df[prices_df["Ticker"] == ticker].copy() if "Ticker" in prices_df.columns else prices_df.copy()
        belief = match_poly_to_ticker(poly_df, ticker, kw_map.get(ticker.upper(), [ticker.lower()]))

        if belief.empty:
            sub["polymarket_prob"] = np.nan
        else:
            # pd.merge_asof requires sorted dates
            sub    = sub.sort_values("Date")
            belief = belief.sort_values("Date")
            merged = pd.merge_asof(
                sub,
                belief.rename(columns={"polymarket_prob": "polymarket_prob"}),
                on="Date",
                direction="nearest",
                tolerance=pd.Timedelta(days=JOIN_TOLERANCE_DAYS),
            )
            sub = merged

        chunks.append(sub)

    result = pd.concat(chunks, ignore_index=True)
    covered = result["polymarket_prob"].notna().mean()
    log.info("Belief join coverage: %.1f%%", covered * 100)
    return result


# ── Feature engineering (B_t + P_t) ──────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    P_t features: returns, volatility, trend
    Applied per ticker.
    """
    df = df.sort_values(["Ticker", "Date"] if "Ticker" in df.columns else ["Date"])
    g = df.groupby("Ticker") if "Ticker" in df.columns else [("ALL", df)]

    chunks = []
    for _, grp in (g if "Ticker" in df.columns else [(None, df)]):
        grp = grp.copy().sort_values("Date")
        c = grp["Close"]

        grp["p_return_1d"]  = c.pct_change(1)
        grp["p_return_5d"]  = c.pct_change(5)
        grp["p_volatility"] = c.pct_change().rolling(20, min_periods=5).std()
        grp["p_trend"]      = c.rolling(20, min_periods=5).mean() / c.rolling(60, min_periods=20).mean() - 1
        grp["p_momentum"]   = c.pct_change(20)
        chunks.append(grp)

    return pd.concat(chunks, ignore_index=True)


def add_belief_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    B_t features: belief_change, belief_volatility, belief_momentum
    Applied per ticker (or globally if no ticker column).
    """
    df = df.copy()
    bp = df["polymarket_prob"]

    if "Ticker" in df.columns:
        chunks = []
        for _, grp in df.groupby("Ticker"):
            grp = grp.copy().sort_values("Date")
            p = grp["polymarket_prob"]
            grp["b_change"]     = p.diff(1)
            grp["b_volatility"] = p.rolling(7, min_periods=3).std()
            grp["b_momentum"]   = p.rolling(7, min_periods=3).mean()
            grp["b_acceleration"] = p.diff(1).diff(1)   # second derivative
            chunks.append(grp)
        return pd.concat(chunks, ignore_index=True)

    df["b_change"]      = bp.diff(1)
    df["b_volatility"]  = bp.rolling(7, min_periods=3).std()
    df["b_momentum"]    = bp.rolling(7, min_periods=3).mean()
    df["b_acceleration"] = bp.diff(1).diff(1)
    return df


# ── Master builder ────────────────────────────────────────────────────────────

def build(
    skip_kalshi: bool = True,
    add_features: bool = True,
) -> pd.DataFrame:
    """
    Main entry point.
    Returns final_dataset DataFrame and saves to CSV.
    """
    log.info("=== BUILD DATASET (polymarket-only mode) ===")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load inputs
    prices  = load_market_prices()
    poly    = load_polymarket_history()
    kalshi  = pd.DataFrame() if skip_kalshi else load_kalshi_history()

    if not skip_kalshi and not kalshi.empty:
        log.info("Kalshi data present but not integrated in this build (reserved for future)")

    # 2. Join belief layer
    final = join_belief_to_prices(prices, poly)

    # 3. Feature engineering
    if add_features:
        final = add_price_features(final)
        final = add_belief_features(final)

    # 4. Sort and clean
    sort_cols = ["Ticker", "Date"] if "Ticker" in final.columns else ["Date"]
    final = final.sort_values(sort_cols).reset_index(drop=True)

    # 5. Save
    final.to_csv(FINAL_DATASET_CSV, index=False)
    log.info(
        "final_dataset.csv: %d rows × %d cols → %s",
        len(final), len(final.columns), FINAL_DATASET_CSV,
    )
    log.info("Columns: %s", list(final.columns))

    # Coverage report
    for col in ["polymarket_prob", "p_return_1d", "b_change"]:
        if col in final.columns:
            pct = final[col].notna().mean() * 100
            log.info("  %-25s  %.1f%% non-null", col, pct)

    return final


def milk_test() -> bool:
    """Quick verification that outputs are non-empty and schema correct."""
    log.info("=== MILK TEST: build_dataset ===")

    if not FINAL_DATASET_CSV.exists():
        log.error("MILK FAIL: final_dataset.csv not found")
        return False

    df = pd.read_csv(FINAL_DATASET_CSV)
    if df.empty:
        log.error("MILK FAIL: final_dataset.csv is empty")
        return False

    required = ["Date", "Close"]
    for col in required:
        if col not in df.columns:
            log.error("MILK FAIL: missing column '%s'", col)
            return False

    log.info("MILK PASS: %d rows, columns: %s", len(df), list(df.columns))
    log.info("Date range: %s → %s", df["Date"].min(), df["Date"].max())
    if "polymarket_prob" in df.columns:
        cov = df["polymarket_prob"].notna().mean()
        log.info("polymarket_prob coverage: %.1f%%", cov * 100)
    return True


if __name__ == "__main__":
    import sys
    if "--milk" in sys.argv:
        ok = milk_test()
        sys.exit(0 if ok else 1)
    df = build()
    print(df.head(5).to_string())
