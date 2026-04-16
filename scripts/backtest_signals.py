"""
backtest_signals.py
===================
Changes from previous version:

1. Uses execution_eligible() from signal_engine to separate signal logging
   from execution PnL. 1d results appear in summary but are flagged LOG_ONLY.

2. Adds gate_rejection_report — shows how many trades each gate filtered,
   so you can see the contribution of each new gate.

3. Output schema unchanged (paper_trade_filtered_overlay_summary.csv).
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

from config import COST_PER_TRADE_BPS, CHAMPION_SYMBOLS, OVERLAY_FLOOR, OVERLAY_CEIL
from signal_engine import (
    infer_signal, champion_filter, execution_eligible,
    load_timeseries_dataset,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR    = Path("data/processed")
SUMMARY_CSV = DATA_DIR / "paper_trade_filtered_overlay_summary.csv"
COSTS_CSV   = DATA_DIR / "paper_trade_filtered_overlay_costs_summary.csv"


def _gate_rejection_report(all_df: pd.DataFrame) -> None:
    """Logs how many trades each gate rejected, for transparency."""
    total = len(all_df)
    passing = all_df["champion_pass"].sum()
    log.info("Gate rejection breakdown (n=%d total):", total)
    reasons = all_df[~all_df["champion_pass"]]["champion_reason"].value_counts()
    for reason, count in reasons.head(10).items():
        pct = count / total * 100
        log.info("  %-50s  n=%4d  (%.1f%%)", reason, count, pct)
    log.info("  → Champion passes: %d (%.1f%%)", passing, passing / total * 100)


def run_backtest(
    df: pd.DataFrame = None,
    output_summary: str = str(SUMMARY_CSV),
    output_costs: str = str(COSTS_CSV),
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if df is None:
        df = load_timeseries_dataset()

    cost = COST_PER_TRADE_BPS / 10_000
    records = []
    prev: dict[str, int] = {}

    for _, row in df.iterrows():
        signal, reason = infer_signal(row)
        r2 = row.copy()
        r2["signal"] = signal
        sym = str(r2.get("Ticker", r2.get("ticker", "")))
        passes, creason = champion_filter(r2, prev_signal=prev.get(sym))
        r2["champion_pass"] = passes
        r2["champion_reason"] = creason
        prev[sym] = signal
        records.append(r2)

    all_df = pd.DataFrame(records)
    _gate_rejection_report(all_df)

    champ = all_df[all_df["champion_pass"]].copy()
    n_champ = len(champ)
    if n_champ == 0:
        log.warning("No champion signals — check thresholds")
        return pd.DataFrame(), pd.DataFrame()

    import re
    horizons = sorted(
        int(m.group(1))
        for col in champ.columns
        if (m := re.match(r"future_return_(\d+)d", col))
    )

    summary_rows = []
    for h in horizons:
        fwd = f"future_return_{h}d"
        if fwd not in champ.columns:
            continue
        sub = champ[fwd].dropna()
        if sub.empty:
            continue
        net = sub - cost
        executes = execution_eligible(h)
        summary_rows.append({
            "horizon":        f"{h}d",
            "execution_mode": "EXECUTE" if executes else "LOG_ONLY",
            "active_signals": len(sub),
            "mean_ret_gross": round(sub.mean(), 6),
            "mean_ret_net":   round(net.mean(), 6),
            "hit_rate_gross": round((sub > 0).mean(), 4),
            "hit_rate_net":   round((net > 0).mean(), 4),
            "std_ret":        round(sub.std(), 6),
            "sharpe_approx":  round(sub.mean() / sub.std() if sub.std() else 0, 3),
            "overlay_floor":  OVERLAY_FLOOR,
            "overlay_ceil":   OVERLAY_CEIL,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Per-symbol 5d breakdown
    fwd5 = "future_return_5d"
    sym_rows = []
    if fwd5 in champ.columns:
        for sym in sorted(CHAMPION_SYMBOLS):
            s = champ[champ["Ticker"] == sym][fwd5].dropna()
            if s.empty:
                continue
            net = s - cost
            sym_rows.append({
                "symbol": sym, "n_signals": len(s),
                "mean_ret_5d": round(s.mean(), 6),
                "mean_net_5d": round(net.mean(), 6),
                "hit_5d":      round((s > 0).mean(), 4),
                "hit_net_5d":  round((net > 0).mean(), 4),
                "overlay_floor": OVERLAY_FLOOR,
            })
    costs_df = pd.DataFrame(sym_rows)

    # Summary log
    log.info("Champion signals: %d  band=[%.2f, %.2f)", n_champ, OVERLAY_FLOOR, OVERLAY_CEIL)
    if not summary_df.empty:
        for _, r in summary_df.iterrows():
            log.info("  %s [%s]: mean_net=%+.4f  hit_net=%.3f  n=%d",
                     r["horizon"], r["execution_mode"],
                     r["mean_ret_net"], r["hit_rate_net"], int(r["active_signals"]))
    if not costs_df.empty:
        log.info("Per-symbol 5d:")
        for _, r in costs_df.iterrows():
            log.info("  %-8s n=%3d  mean_net=%+.4f  hit_net=%.3f",
                     r["symbol"], int(r["n_signals"]), r["mean_net_5d"], r["hit_net_5d"])

    os.makedirs(DATA_DIR, exist_ok=True)
    summary_df.to_csv(output_summary, index=False)
    costs_df.to_csv(output_costs, index=False)
    log.info("Summary → %s", output_summary)
    log.info("Costs   → %s", output_costs)
    return summary_df, costs_df


def milk_test() -> bool:
    log.info("=== MILK TEST: backtest_signals ===")
    np.random.seed(2024)
    n = 300
    syms    = np.random.choice(["GLD", "^GDAXI", "USO", "SPY"], n)
    overlay = np.random.uniform(0.50, 0.80, n)
    lags    = np.random.choice([-60, 30, 60], n, p=[0.3, 0.3, 0.4])
    vol_20d = np.random.uniform(0.006, 0.035, n)
    trend   = np.random.uniform(-0.02, 0.02, n)
    mom     = np.random.uniform(-0.03, 0.03, n)

    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Ticker": syms,
        "match_score": np.random.uniform(0.3, 0.9, n),
        "lead_lag_minutes": lags.astype(float),
        "t0_volatility_20d": vol_20d,
        "t0_trend_50_200": trend,
        "t0_momentum_20d": mom,
        "overlay_score": overlay,
        "future_return_1d":  np.random.randn(n) * 0.01,
        "future_return_5d":  np.random.randn(n) * 0.02,
        "future_return_10d": np.random.randn(n) * 0.03,
    })
    # Inject edge in the good zone
    good = (
        np.isin(syms, ["GLD", "^GDAXI", "USO"]) &
        (overlay >= 0.60) & (overlay < 0.75) &
        (lags > 0) & (vol_20d >= 0.012) &
        ((trend >= 0) | (mom >= 0))
    )
    df.loc[good, "future_return_5d"] += 0.012

    summary, costs = run_backtest(df,
                                  output_summary="/tmp/test_summary.csv",
                                  output_costs="/tmp/test_costs.csv")
    ok = True

    # 1. Summary non-empty
    if summary.empty:
        log.error("FAIL: empty summary")
        ok = False

    # 2. 1d must be LOG_ONLY
    if not summary.empty and "1d" in summary["horizon"].values:
        r1d = summary[summary["horizon"] == "1d"].iloc[0]
        if r1d["execution_mode"] != "LOG_ONLY":
            log.error("FAIL: 1d should be LOG_ONLY, got %s", r1d["execution_mode"])
            ok = False
        else:
            log.info("PASS ✓  1d horizon correctly marked LOG_ONLY")

    # 3. 5d must be EXECUTE
    if not summary.empty and "5d" in summary["horizon"].values:
        r5d = summary[summary["horizon"] == "5d"].iloc[0]
        if r5d["execution_mode"] != "EXECUTE":
            log.error("FAIL: 5d should be EXECUTE, got %s", r5d["execution_mode"])
            ok = False
        else:
            log.info("PASS ✓  5d horizon correctly marked EXECUTE  mean_net=%+.4f  hit_net=%.3f",
                     r5d["mean_ret_net"], r5d["hit_rate_net"])

    # 4. New floor: no signals below 0.60 should pass
    from signal_engine import champion_filter
    test_row = pd.Series({
        "signal": 1, "Ticker": "GLD", "overlay_score": 0.58,
        "match_score": 0.7, "lead_lag_minutes": 30.0,
        "t0_volatility_20d": 0.020, "t0_trend_50_200": 0.01, "t0_momentum_20d": 0.01,
    })
    passes, _ = champion_filter(test_row)
    if passes:
        log.error("FAIL: overlay 0.58 should be REJECTED with floor=0.60")
        ok = False
    else:
        log.info("PASS ✓  overlay 0.58 correctly REJECTED (floor=0.60)")

    # 5. Vol gate: low vol should be rejected
    test_low_vol = test_row.copy()
    test_low_vol["overlay_score"] = 0.65
    test_low_vol["t0_volatility_20d"] = 0.008
    passes2, _ = champion_filter(test_low_vol)
    if passes2:
        log.error("FAIL: vol=0.008 should be REJECTED")
        ok = False
    else:
        log.info("PASS ✓  vol=0.008 correctly REJECTED (MIN_VOL=0.012)")

    # 6. Trend gate: counter-trend should be rejected
    test_ct = test_row.copy()
    test_ct["overlay_score"] = 0.65
    test_ct["t0_trend_50_200"] = -0.015
    test_ct["t0_momentum_20d"] = -0.020
    passes3, _ = champion_filter(test_ct)
    if passes3:
        log.error("FAIL: counter-trend should be REJECTED")
        ok = False
    else:
        log.info("PASS ✓  counter-trend correctly REJECTED (trend=-0.015, mom=-0.020)")

    log.info("MILK TEST: %s", "ALL PASS ✓" if ok else "FAILURES DETECTED")
    return ok


if __name__ == "__main__":
    import sys
    if "--milk" in sys.argv:
        sys.exit(0 if milk_test() else 1)
    run_backtest()
