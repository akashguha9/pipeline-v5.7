"""
signal_engine.py
================
Minimal changes from previous version:

1.  champion_filter() — two new gates added, zero architecture change:
    Gate 4 (NEW): volatility regime gate (MIN_VOL_20D)
    Gate 5 (NEW): trend confirmation gate (REQUIRE_TREND_CONFIRMATION)

2.  execution_eligible() helper — NEW: separates signal generation from
    execution decision. Signals at horizons < MIN_HORIZON_FOR_EXECUTION
    are logged (champion_pass=True) but not executed (execute=False).
    This suppresses cost-negative 1d trades without removing the signal.

3.  infer_signal() — UNCHANGED

4.  All threshold constants imported from config — no hardcoded values.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

try:
    from config import (
        MIN_SIGNAL_MATCH_SCORE,
        OVERLAY_FLOOR,
        OVERLAY_CEIL,
        CHAMPION_SYMBOLS,
        LONG_ONLY,
        MIN_SIGNAL_PERSISTENCE_DAYS,
        MIN_VOL_20D,
        REQUIRE_TREND_CONFIRMATION,
        MIN_HORIZON_FOR_EXECUTION,
        COST_PER_TRADE_BPS,
    )
except ImportError:
    MIN_SIGNAL_MATCH_SCORE      = 0.30
    OVERLAY_FLOOR               = 0.60
    OVERLAY_CEIL                = 0.75
    CHAMPION_SYMBOLS            = {"GLD", "^GDAXI", "USO"}
    LONG_ONLY                   = True
    MIN_SIGNAL_PERSISTENCE_DAYS = 1
    MIN_VOL_20D                 = 0.012
    REQUIRE_TREND_CONFIRMATION  = True
    MIN_HORIZON_FOR_EXECUTION   = 5
    COST_PER_TRADE_BPS          = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "processed"


# ── Core signal inference (UNCHANGED) ─────────────────────────────────────────

def infer_signal(row: pd.Series) -> tuple[int, str]:
    score = row.get("match_score")
    lag   = row.get("lead_lag_minutes")
    vol   = row.get("t0_volatility_20d")
    trend = row.get("t0_trend_50_200")
    mom20 = row.get("t0_momentum_20d")

    signal = 0
    reason = "no_signal"

    if pd.isna(score) or pd.isna(lag):
        return signal, "data_void"

    if score >= MIN_SIGNAL_MATCH_SCORE and lag > 0:
        signal = 1
        reason = "news_led"
    elif score >= MIN_SIGNAL_MATCH_SCORE and lag < 0:
        signal = -1
        reason = "market_led"

    if pd.notna(vol)   and vol   > 0.04:  reason += "_high_vol"
    if pd.notna(trend) and trend > 0:     reason += "_uptrend"
    elif pd.notna(trend):                 reason += "_downtrend"
    if pd.notna(mom20) and mom20 > 0:     reason += "_mom_up"
    elif pd.notna(mom20):                 reason += "_mom_down"

    return signal, reason


# ── Champion filter (refined with 2 new gates) ────────────────────────────────

def champion_filter(
    row: pd.Series,
    prev_signal: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Champion filter with 5 ordered gates.

    Gate 1: Long-only         (unchanged, fail-fast)
    Gate 2: Symbol universe   (unchanged)
    Gate 3: Overlay band      (OVERLAY_FLOOR now 0.60, was 0.55)
    Gate 4: Vol regime        (NEW: t0_volatility_20d >= MIN_VOL_20D)
    Gate 5: Trend confirm     (NEW: at least one of trend/momentum positive)
    Gate 6: Persistence       (optional, MIN_SIGNAL_PERSISTENCE_DAYS)
    """
    # Gate 1: LONG ONLY (fail-fast)
    if LONG_ONLY:
        signal_val = int(row.get("signal", row.get("inferred_signal", 0)))
        if signal_val != 1:
            return False, "rejected_not_long"

    # Gate 2: Symbol
    symbol = str(row.get("Ticker", row.get("ticker", row.get("symbol", ""))))
    if symbol not in CHAMPION_SYMBOLS:
        return False, f"rejected_symbol_{symbol}"

    # Gate 3: Overlay band
    overlay = row.get("overlay_score", row.get("composite_score", np.nan))
    if pd.isna(overlay):
        return False, "rejected_no_overlay"
    abs_ov = abs(float(overlay))
    if abs_ov < OVERLAY_FLOOR:
        return False, f"rejected_below_floor_{abs_ov:.3f}"
    if abs_ov >= OVERLAY_CEIL:
        return False, f"rejected_above_ceil_{abs_ov:.3f}"

    # Gate 4: Volatility regime (NEW)
    if MIN_VOL_20D is not None:
        vol_20d = row.get("t0_volatility_20d", np.nan)
        if pd.notna(vol_20d) and float(vol_20d) < MIN_VOL_20D:
            return False, f"rejected_low_vol_{float(vol_20d):.4f}"
        # If vol column is missing (NaN), we pass through — don't reject on absent data

    # Gate 5: Trend confirmation (NEW)
    if REQUIRE_TREND_CONFIRMATION:
        trend = row.get("t0_trend_50_200", np.nan)
        mom   = row.get("t0_momentum_20d", np.nan)
        # Need at least one non-NaN value; require it to be non-negative
        has_trend = pd.notna(trend)
        has_mom   = pd.notna(mom)
        if has_trend or has_mom:
            trend_ok = has_trend and float(trend) >= 0
            mom_ok   = has_mom   and float(mom)   >= 0
            if not (trend_ok or mom_ok):
                return False, f"rejected_counter_trend_t{float(trend) if has_trend else 'NA':.4f}_m{float(mom) if has_mom else 'NA':.4f}"
        # If both trend and momentum are NaN, pass through — don't penalise missing data

    # Gate 6: Persistence (optional)
    if MIN_SIGNAL_PERSISTENCE_DAYS >= 2 and prev_signal is not None:
        if int(prev_signal) != 1:
            return False, "rejected_no_persistence"

    return True, f"champion_{symbol}_{abs_ov:.3f}"


# ── Execution eligibility (NEW helper) ────────────────────────────────────────

def execution_eligible(horizon_days: int) -> bool:
    """
    Returns True if the forward return horizon meets the minimum for execution.

    Evidence: hit_1d_net = 0.498 (coin flip after costs). Only 5d+ is worth
    executing. 1d signals pass champion_filter but are NOT executed.
    This separates signal logging from position taking.
    """
    return horizon_days >= MIN_HORIZON_FOR_EXECUTION


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_timeseries_dataset(path: Optional[str] = None) -> pd.DataFrame:
    candidates = [
        path,
        str(DATA_DIR / "output_timeseries_dataset.csv"),
        str(DATA_DIR / "final_dataset.csv"),
        "output_timeseries_dataset.csv",
        "data/processed/output_timeseries_dataset.csv",
    ]
    for p in candidates:
        if p and Path(p).exists():
            df = pd.read_csv(p)
            log.info("Loaded %s (%d rows)", p, len(df))
            return df
    raise FileNotFoundError(f"Dataset not found. Tried: {[c for c in candidates if c]}")


def extract_horizons(df: pd.DataFrame) -> list[int]:
    import re
    return sorted(
        int(m.group(1))
        for col in df.columns
        if (m := re.match(r"future_return_(\d+)d", col))
    )


# ── Main runner ────────────────────────────────────────────────────────────────

def run_signal_engine(
    df: Optional[pd.DataFrame] = None,
    output_path: str = "output_signals.csv",
) -> pd.DataFrame:
    if df is None:
        df = load_timeseries_dataset()

    horizons = extract_horizons(df)
    log.info("Horizons: %s", horizons)
    log.info(
        "Champion rule: LONG=%s syms=%s band=[%.2f,%.2f) vol>=%.3f trend_req=%s",
        LONG_ONLY, CHAMPION_SYMBOLS, OVERLAY_FLOOR, OVERLAY_CEIL,
        MIN_VOL_20D or 0, REQUIRE_TREND_CONFIRMATION,
    )

    rows          = []
    prev_signals: dict[str, int] = {}

    for _, row in df.iterrows():
        signal, sig_reason = infer_signal(row)
        row = row.copy()
        row["signal"]        = signal
        row["signal_reason"] = sig_reason

        sym = str(row.get("Ticker", row.get("ticker", "")))
        passes, champ_reason = champion_filter(row, prev_signal=prev_signals.get(sym))
        row["champion_pass"]   = passes
        row["champion_reason"] = champ_reason
        prev_signals[sym] = signal
        rows.append(row)

    out = pd.DataFrame(rows)
    total  = len(out)
    active = out["champion_pass"].sum()
    log.info("Total: %d  Champion: %d (%.1f%%)", total, active, active/total*100)

    if horizons and active > 0:
        champ = out[out["champion_pass"]]
        cost  = COST_PER_TRADE_BPS / 10_000
        for h in horizons:
            fwd = f"future_return_{h}d"
            executes = execution_eligible(h)
            if fwd in champ.columns:
                sub = champ[fwd].dropna()
                if len(sub):
                    net = sub - cost
                    exec_flag = "EXECUTE" if executes else "LOG_ONLY"
                    log.info(
                        "  %2dd [%s]: mean_gross=%+.4f  mean_net=%+.4f  hit_gross=%.3f  hit_net=%.3f  n=%d",
                        h, exec_flag, sub.mean(), net.mean(), (sub > 0).mean(), (net > 0).mean(), len(sub),
                    )

    out.to_csv(output_path, index=False)
    log.info("Saved → %s", output_path)
    return out


# ── Milk test ──────────────────────────────────────────────────────────────────

def milk_test() -> bool:
    log.info("=== MILK TEST: signal_engine (recalibrated gates) ===")
    ok = True

    def row(signal, sym, overlay, vol=0.020, trend=0.01, mom=0.01):
        return pd.Series({
            "signal": signal, "Ticker": sym,
            "overlay_score": overlay,
            "match_score": 0.65, "lead_lag_minutes": 30.0,
            "t0_volatility_20d": vol,
            "t0_trend_50_200": trend,
            "t0_momentum_20d": mom,
        })

    tests = [
        # description, row_args, expected_pass
        # ── Floor change verification ──────────────────────────────
        ("overlay 0.57 → REJECTED (below new floor 0.60)",
         row(1,"GLD",0.57), False),
        ("overlay 0.59 → REJECTED (below new floor 0.60)",
         row(1,"GLD",0.59), False),
        ("overlay 0.60 → PASSES (at new floor)",
         row(1,"GLD",0.60), True),
        # ── Volatility gate ────────────────────────────────────────
        ("vol=0.009 → REJECTED (below MIN_VOL_20D=0.012)",
         row(1,"GLD",0.65, vol=0.009), False),
        ("vol=0.012 → PASSES (at MIN_VOL_20D)",
         row(1,"GLD",0.65, vol=0.012), True),
        ("vol=0.020 → PASSES (above MIN_VOL_20D)",
         row(1,"GLD",0.65, vol=0.020), True),
        ("vol=NaN → PASSES (missing data, don't penalise)",
         row(1,"GLD",0.65, vol=float("nan")), True),
        # ── Trend gate ─────────────────────────────────────────────
        ("trend=-0.01,mom=-0.02 → REJECTED (counter-trend)",
         row(1,"GLD",0.65, trend=-0.01, mom=-0.02), False),
        ("trend=+0.01,mom=-0.02 → PASSES (trend positive, OR gate)",
         row(1,"GLD",0.65, trend=0.01, mom=-0.02), True),
        ("trend=-0.01,mom=+0.01 → PASSES (momentum positive, OR gate)",
         row(1,"GLD",0.65, trend=-0.01, mom=0.01), True),
        ("trend=NaN,mom=NaN → PASSES (missing, don't penalise)",
         row(1,"GLD",0.65, trend=float("nan"), mom=float("nan")), True),
        # ── Existing gates (must still work) ──────────────────────
        ("short signal → REJECTED",
         row(-1,"GLD",0.65), False),
        ("non-champion symbol SPY → REJECTED",
         row(1,"SPY",0.65), False),
        ("overlay >= 0.75 → REJECTED",
         row(1,"GLD",0.75), False),
    ]

    for desc, r, expected in tests:
        passes, reason = champion_filter(r)
        if passes != expected:
            log.error("FAIL ✗  %s | got=%s expected=%s | reason=%s",
                      desc, passes, expected, reason)
            ok = False
        else:
            log.info("PASS ✓  %s", desc)

    # Execution eligibility tests
    log.info("--- Execution eligibility ---")
    for h, expected in [(1, False), (3, False), (5, True), (10, True)]:
        result = execution_eligible(h)
        if result != expected:
            log.error("FAIL ✗  horizon=%dd eligible=%s expected=%s", h, result, expected)
            ok = False
        else:
            log.info("PASS ✓  horizon=%dd → %s", h, "EXECUTE" if result else "LOG_ONLY")

    if ok:
        log.info("MILK TEST: ALL PASS ✓")
        log.info("Key: floor is now 0.60, vol gate active, trend gate active, 1d suppressed")
    else:
        log.error("MILK TEST: FAILURES DETECTED")
    return ok


if __name__ == "__main__":
    import sys
    if "--milk" in sys.argv:
        sys.exit(0 if milk_test() else 1)
    run_signal_engine()
