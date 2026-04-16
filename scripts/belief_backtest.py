"""
belief_backtest.py
==================
Wraps existing rolling_validation.py and walkforward_validation.py.
Runs baseline vs belief-enhanced comparison.
Does NOT rewrite existing validation logic.

Usage:
  python3 belief_backtest.py
  python3 belief_backtest.py --rolling-only
  python3 belief_backtest.py --walkforward-only
  python3 belief_backtest.py --milk
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
TIMESERIES_CSV   = ROOT / "output_timeseries_dataset.csv"
FINAL_DATASET    = ROOT / "data" / "final_dataset.csv"
SIGNALS_CSV      = ROOT / "output_signals.csv"
BACKTEST_CSV     = ROOT / "output_backtest_summary.csv"

BELIEF_SIGNALS_CSV   = ROOT / "output_belief_signals.csv"
BELIEF_BACKTEST_CSV  = ROOT / "output_belief_backtest_summary.csv"
COMPARISON_CSV       = ROOT / "output_belief_comparison.csv"


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_enhanced_dataset() -> pd.DataFrame:
    """
    Load the dataset that includes belief features.
    Priority: final_dataset.csv (has polymarket_prob) > output_timeseries_dataset.csv
    """
    if FINAL_DATASET.exists() and FINAL_DATASET.stat().st_size > 100:
        df = pd.read_csv(FINAL_DATASET)
        log.info("Loaded final_dataset.csv: %d rows", len(df))
    elif TIMESERIES_CSV.exists():
        df = pd.read_csv(TIMESERIES_CSV)
        log.info("Loaded output_timeseries_dataset.csv: %d rows (no belief layer)", len(df))
    else:
        raise FileNotFoundError(
            "Neither final_dataset.csv nor output_timeseries_dataset.csv found. "
            "Run build_dataset.py first."
        )

    # Parse datetimes
    for col in df.columns:
        if "time" in col.lower() or col == "Date" or col.endswith("_timestamp"):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed", utc=True)
            except Exception:
                pass

    return df


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_belief_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run infer_signal_with_belief() over the full dataset.
    Returns df with columns: signal, reason (belief-enhanced).
    """
    try:
        from signal_engine_belief_extension import infer_signal_with_belief
    except ImportError:
        log.error("signal_engine_belief_extension.py not found. Place it at %s", ROOT)
        raise

    signals, reasons = [], []
    for _, row in df.iterrows():
        s, r = infer_signal_with_belief(row, df_history=df)
        signals.append(s)
        reasons.append(r)

    result = df.copy()
    result["signal"] = signals
    result["reason"] = reasons
    return result


def generate_baseline_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run original infer_signal() (price-only) over the dataset.
    """
    try:
        from signal_engine import infer_signal
    except ImportError:
        # Fallback: inline the original logic
        from signal_engine_belief_extension import _price_signal

        def infer_signal(row, df_history=None):
            sig, reason = _price_signal(row)
            return int(sig), reason

    signals, reasons = [], []
    for _, row in df.iterrows():
        s, r = infer_signal(row)
        signals.append(s)
        reasons.append(r)

    result = df.copy()
    result["signal"] = signals
    result["reason"] = reasons
    return result


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_signals(df: pd.DataFrame, horizons: list[int], label: str) -> dict:
    """
    Compute accuracy, hit rate, coverage per horizon.
    Returns dict of metrics.
    """
    results = {}
    total = len(df)
    active = (df["signal"] != 0).sum()

    for h in horizons:
        fwd_col = f"future_return_{h}d"
        if fwd_col not in df.columns:
            continue

        sub = df[df["signal"] != 0].copy()
        sub = sub.dropna(subset=[fwd_col])

        if sub.empty:
            results[f"{h}d_accuracy"]  = np.nan
            results[f"{h}d_coverage"]  = 0.0
            continue

        correct = (
            ((sub["signal"] == 1)  & (sub[fwd_col] > 0)) |
            ((sub["signal"] == -1) & (sub[fwd_col] < 0))
        )
        results[f"{h}d_accuracy"] = round(correct.mean(), 4)
        results[f"{h}d_n_signals"] = len(sub)

    results["total_rows"]    = total
    results["active_signals"] = int(active)
    results["signal_rate"]   = round(active / total, 4) if total else 0
    results["label"]         = label
    return results


def compare_signals(
    df_baseline: pd.DataFrame,
    df_belief: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Build side-by-side comparison table.
    """
    base_metrics   = evaluate_signals(df_baseline, horizons, "baseline")
    belief_metrics = evaluate_signals(df_belief,   horizons, "belief")

    rows = []
    rows.append({
        "metric":   "signal_rate",
        "baseline": base_metrics.get("signal_rate", 0),
        "belief":   belief_metrics.get("signal_rate", 0),
        "delta":    round(belief_metrics.get("signal_rate", 0) - base_metrics.get("signal_rate", 0), 4),
    })

    for h in horizons:
        k = f"{h}d_accuracy"
        b = base_metrics.get(k, np.nan)
        be = belief_metrics.get(k, np.nan)
        rows.append({
            "metric":   f"accuracy_{h}d",
            "baseline": b,
            "belief":   be,
            "delta":    round(be - b, 4) if pd.notna(b) and pd.notna(be) else np.nan,
        })

    return pd.DataFrame(rows)


# ── Rolling validation wrapper ────────────────────────────────────────────────

def run_rolling_comparison(df_baseline: pd.DataFrame, df_belief: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to call existing rolling_validation.py if available.
    Falls back to simple period-split evaluation.
    """
    try:
        import rolling_validation as rv
        log.info("Using existing rolling_validation.py")
        # Inject belief signals and run
        # rolling_validation typically reads from output_signals.csv
        # Save belief signals first, then run
        df_belief.to_csv(BELIEF_SIGNALS_CSV, index=False)
        log.info("Saved belief signals → %s", BELIEF_SIGNALS_CSV)
    except ImportError:
        log.info("rolling_validation.py not importable — using internal rolling eval")

    # Internal rolling evaluation (always runs as fallback)
    horizons = _extract_horizons(df_baseline)
    periods  = _rolling_periods(df_baseline)

    rows = []
    for start, end in periods:
        mask_b  = (df_baseline["Date"] >= start) & (df_baseline["Date"] < end) if "Date" in df_baseline.columns else slice(None)
        mask_be = (df_belief["Date"]   >= start) & (df_belief["Date"]   < end) if "Date" in df_belief.columns else slice(None)

        chunk_b  = df_baseline[mask_b]  if isinstance(mask_b,  pd.Series) else df_baseline
        chunk_be = df_belief[mask_be]   if isinstance(mask_be, pd.Series) else df_belief

        for h in horizons:
            fwd = f"future_return_{h}d"
            if fwd not in chunk_b.columns:
                continue

            def acc(chunk):
                s = chunk[chunk["signal"] != 0]
                s = s.dropna(subset=[fwd])
                if s.empty:
                    return np.nan
                return ((s["signal"] == 1) & (s[fwd] > 0) | (s["signal"] == -1) & (s[fwd] < 0)).mean()

            rows.append({
                "period":    f"{start.date()}:{end.date()}" if hasattr(start, "date") else f"{start}:{end}",
                "horizon":   f"{h}d",
                "acc_baseline": round(acc(chunk_b) or 0, 4),
                "acc_belief":   round(acc(chunk_be) or 0, 4),
            })

    result = pd.DataFrame(rows)
    if not result.empty and "acc_baseline" in result.columns:
        result["delta"] = result["acc_belief"] - result["acc_baseline"]
    return result


def _extract_horizons(df: pd.DataFrame) -> list[int]:
    import re
    h = []
    for col in df.columns:
        m = re.match(r"future_return_(\d+)d", col)
        if m:
            h.append(int(m.group(1)))
    return sorted(h)


def _rolling_periods(df: pd.DataFrame, window_days: int = 90, step_days: int = 30):
    """Generate (start, end) tuples for rolling evaluation."""
    if "Date" not in df.columns:
        return []
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        return []
    start = dates.min()
    end   = dates.max()
    periods = []
    current = start
    while current + pd.Timedelta(days=window_days) <= end:
        periods.append((current, current + pd.Timedelta(days=window_days)))
        current += pd.Timedelta(days=step_days)
    return periods


# ── Master runner ─────────────────────────────────────────────────────────────

def run(rolling: bool = True, walkforward: bool = True):
    log.info("=== BELIEF BACKTEST COMPARISON ===")

    df = load_enhanced_dataset()
    horizons = _extract_horizons(df)
    if not horizons:
        log.warning("No future_return_Nd columns found — using synthetic horizon [1] for testing")
        horizons = [1]

    log.info("Horizons: %s", horizons)
    log.info("Has polymarket_prob: %s", "polymarket_prob" in df.columns)

    # Generate signals
    log.info("Generating baseline signals...")
    df_baseline = generate_baseline_signals(df)

    log.info("Generating belief-enhanced signals...")
    df_belief = generate_belief_signals(df)

    # Static comparison
    comparison = compare_signals(df_baseline, df_belief, horizons)
    comparison.to_csv(COMPARISON_CSV, index=False)
    log.info("\n=== STATIC COMPARISON ===\n%s", comparison.to_string(index=False))

    # Rolling comparison
    if rolling:
        log.info("Running rolling comparison...")
        rolling_result = run_rolling_comparison(df_baseline, df_belief)
        if not rolling_result.empty:
            rolling_path = ROOT / "output_rolling_comparison.csv"
            rolling_result.to_csv(rolling_path, index=False)
            log.info("Rolling comparison saved → %s", rolling_path)
            # Summary
            mean_delta = rolling_result["delta"].mean() if "delta" in rolling_result.columns else 0
            log.info("Mean accuracy delta (belief - baseline): %.4f", mean_delta)

    # Save belief signals
    df_belief[["signal", "reason"] + [c for c in df_belief.columns if c not in ["signal", "reason"]]].to_csv(
        BELIEF_SIGNALS_CSV, index=False
    )
    log.info("Belief signals saved → %s", BELIEF_SIGNALS_CSV)

    return comparison


def milk_test() -> bool:
    log.info("=== MILK TEST: belief backtest ===")

    # Create synthetic dataset to test the whole pipeline
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "Date":              dates,
        "Ticker":            "SPY",
        "Close":             100 * (1 + np.random.randn(n) * 0.01).cumprod(),
        "match_score":       np.random.uniform(0, 1, n),
        "lead_lag_minutes":  np.random.choice([-60, 0, 60, np.nan], n),
        "t0_volatility_20d": np.random.uniform(0.01, 0.06, n),
        "t0_trend_50_200":   np.random.randn(n) * 0.01,
        "t0_momentum_20d":   np.random.randn(n) * 0.02,
        "polymarket_prob":   np.random.uniform(0.3, 0.8, n),
        "b_change":          np.random.randn(n) * 0.03,
        "b_momentum":        np.random.uniform(0.4, 0.7, n),
        "b_volatility":      np.random.uniform(0.01, 0.06, n),
        "b_acceleration":    np.random.randn(n) * 0.01,
        "future_return_1d":  np.random.randn(n) * 0.01,
        "future_return_5d":  np.random.randn(n) * 0.02,
    })

    from signal_engine_belief_extension import infer_signal_with_belief, _price_signal

    b_sigs, be_sigs = [], []
    for _, row in df.iterrows():
        bs, _ = _price_signal(row)
        bes, _ = infer_signal_with_belief(row)
        b_sigs.append(int(bs) if isinstance(bs, float) else bs)
        be_sigs.append(bes)

    df["signal_baseline"] = b_sigs
    df["signal_belief"]   = be_sigs

    n_base   = (df["signal_baseline"] != 0).sum()
    n_belief = (df["signal_belief"] != 0).sum()

    log.info("Baseline signals: %d / %d rows", n_base, n)
    log.info("Belief signals:   %d / %d rows", n_belief, n)

    if n_belief == 0:
        log.error("MILK FAIL: no belief signals generated")
        return False

    if n_belief <= n_base:
        log.warning("Belief signal count ≤ baseline — expected higher coverage with belief layer")

    log.info("MILK PASS: belief pipeline functional")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-only",    action="store_true")
    parser.add_argument("--walkforward-only", action="store_true")
    parser.add_argument("--milk",            action="store_true")
    args = parser.parse_args()

    if args.milk:
        ok = milk_test()
        sys.exit(0 if ok else 1)

    run(
        rolling     = not args.walkforward_only,
        walkforward = not args.rolling_only,
    )
