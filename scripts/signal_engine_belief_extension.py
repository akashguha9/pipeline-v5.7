"""
signal_engine_belief_extension.py
===================================
Minimal extension to signal_engine.py's infer_signal().

DO NOT rewrite signal_engine.py from scratch.
This module provides:
  1. infer_signal_with_belief() — drop-in replacement for infer_signal()
     that adds belief-layer signals on top of price signals
  2. S_t = α * price_signal + β * belief_signal

Integration:
  In signal_engine.py, add ONE import at the top:
    from signal_engine_belief_extension import infer_signal_with_belief as infer_signal
  
  That's the ONLY change to signal_engine.py.
  Everything downstream (evaluate_signal, groupby, CSV output) remains untouched.

Signal formula:
  price_signal: existing lead-lag logic (news_led / market_led)
  belief_signal: polymarket_prob derivatives
  combined: α=0.6 * price_signal + β=0.4 * belief_signal
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# These must stay consistent with signal_engine.py
MIN_SIGNAL_MATCH_SCORE = 0.3   # fallback if config not importable

try:
    from config import MIN_SIGNAL_MATCH_SCORE
except ImportError:
    pass

# Weights for combined signal
ALPHA_PRICE   = 0.6   # weight on price/news signal
BETA_BELIEF   = 0.4   # weight on belief signal

# Thresholds for belief signal
BELIEF_CHANGE_THRESHOLD     = 0.02   # |Δprob| > 2% → directional signal
BELIEF_MOMENTUM_THRESHOLD   = 0.50   # prob > 50% → bullish regime
BELIEF_VOLATILITY_HIGH      = 0.05   # std > 5% → uncertain


def _price_signal(row: pd.Series, min_score: float = MIN_SIGNAL_MATCH_SCORE) -> tuple[float, str]:
    """
    Exact replica of original infer_signal() logic.
    Returns (signal_float, reason_str).
    signal_float ∈ {-1.0, 0.0, 1.0}
    """
    score = row.get("match_score")
    lag   = row.get("lead_lag_minutes")
    vol   = row.get("t0_volatility_20d")
    trend = row.get("t0_trend_50_200")
    mom20 = row.get("t0_momentum_20d")

    signal = 0.0
    reason = "no_signal"

    if pd.isna(score) or pd.isna(lag):
        # DATA VOID — handled by data_void_engine if integrated, else zero
        return signal, "data_void"

    if score >= min_score and lag > 0:
        signal = 1.0
        reason = "news_led"
    elif score >= min_score and lag < 0:
        signal = -1.0
        reason = "market_led"

    if pd.notna(vol) and vol > 0.04:
        reason += "_high_vol"
    if pd.notna(trend):
        reason += "_uptrend" if trend > 0 else "_downtrend"
    if pd.notna(mom20):
        reason += "_mom_up" if mom20 > 0 else "_mom_down"

    return signal, reason


def _belief_signal(row: pd.Series) -> tuple[float, str]:
    """
    Derive signal from B_t features.
    Returns (signal_float, reason_str).
    signal_float ∈ [-1.0, 1.0] (continuous, not just ±1)
    """
    prob    = row.get("polymarket_prob")
    change  = row.get("b_change")       # Δprob
    mom     = row.get("b_momentum")     # rolling mean
    vol     = row.get("b_volatility")   # rolling std
    accel   = row.get("b_acceleration") # second derivative

    # All NaN → no belief signal
    if pd.isna(prob) and pd.isna(change) and pd.isna(mom):
        return 0.0, "belief_void"

    score  = 0.0
    parts  = []

    # Level signal: high prob = bullish on the underlying event resolving YES
    if pd.notna(prob):
        if prob > 0.65:
            score += 0.5
            parts.append("prob_high")
        elif prob < 0.35:
            score -= 0.5
            parts.append("prob_low")

    # Momentum signal: rising belief = bullish
    if pd.notna(change) and abs(change) > BELIEF_CHANGE_THRESHOLD:
        score += np.sign(change) * 0.4
        parts.append("b_rising" if change > 0 else "b_falling")

    # Momentum confirmation: rolling mean direction
    if pd.notna(mom):
        if mom > BELIEF_MOMENTUM_THRESHOLD + 0.05:
            score += 0.15
            parts.append("b_mom_bull")
        elif mom < BELIEF_MOMENTUM_THRESHOLD - 0.05:
            score -= 0.15
            parts.append("b_mom_bear")

    # Acceleration: second derivative confirmation
    if pd.notna(accel) and abs(accel) > 0.005:
        score += np.sign(accel) * 0.1
        parts.append("b_accel_up" if accel > 0 else "b_accel_dn")

    # Uncertainty penalty: high volatility compresses signal
    if pd.notna(vol) and vol > BELIEF_VOLATILITY_HIGH:
        score *= 0.7
        parts.append("b_uncertain")

    # Clip to [-1, 1]
    score = float(np.clip(score, -1.0, 1.0))
    reason = "belief_" + "_".join(parts) if parts else "belief_neutral"
    return score, reason


def infer_signal_with_belief(
    row: pd.Series,
    df_history: Optional[pd.DataFrame] = None,
    min_score: float = MIN_SIGNAL_MATCH_SCORE,
    alpha: float = ALPHA_PRICE,
    beta:  float = BETA_BELIEF,
) -> tuple[int, str]:
    """
    Combined signal: S_t = α * price_signal + β * belief_signal

    Drop-in replacement for signal_engine.infer_signal().
    Returns (signal: int, reason: str) — same interface.
    
    INTEGRATION:
    In signal_engine.py, replace:
      from config import MIN_SIGNAL_MATCH_SCORE
    with:
      from signal_engine_belief_extension import infer_signal_with_belief as infer_signal
      from config import MIN_SIGNAL_MATCH_SCORE
    
    And in the main loop, change:
      signal, reason = infer_signal(row)
    to:
      signal, reason = infer_signal(row, df_history=df)
    """
    p_sig, p_reason = _price_signal(row, min_score)
    b_sig, b_reason = _belief_signal(row)

    # If price signal is data void, attempt belief-only fallback
    # (data_void_engine handles this more formally, but belief provides
    #  a cleaner path when polymarket data is present)
    if p_reason == "data_void":
        if abs(b_sig) > 0.3:
            combined = b_sig
            final_reason = f"belief_only_{b_reason}"
        else:
            # Fall through to data_void_engine if available
            try:
                from data_void_engine import infer_signal_with_void_fallback
                return infer_signal_with_void_fallback(row, df_history=df_history, min_score=min_score)
            except ImportError:
                return 0, "no_signal_void"
    else:
        # Standard combination
        combined = alpha * p_sig + beta * b_sig

    # Convert continuous score to discrete signal
    if combined > 0.15:
        signal = 1
    elif combined < -0.15:
        signal = -1
    else:
        signal = 0

    # Enrich reason string
    reason_parts = [p_reason]
    if b_reason not in ("belief_void", "belief_neutral"):
        reason_parts.append(b_reason)
    reason = "__".join(reason_parts)

    return signal, reason


# ── Backtest comparison utility ───────────────────────────────────────────────

def compare_baseline_vs_belief(
    df: pd.DataFrame,
    horizons: list[int] = None,
) -> pd.DataFrame:
    """
    Runs both original and belief-enhanced signals on df.
    Returns comparison DataFrame showing accuracy improvement per seed/horizon.

    Usage:
      df = pd.read_csv("output_timeseries_dataset.csv")
      comparison = compare_baseline_vs_belief(df)
      print(comparison)
    """
    import re

    if horizons is None:
        horizons = []
        for col in df.columns:
            m = re.match(r"future_return_(\d+)d", col)
            if m:
                horizons.append(int(m.group(1)))
        horizons = sorted(horizons)

    baseline_signals = []
    belief_signals   = []

    for _, row in df.iterrows():
        bs, _ = _price_signal(row)
        bls, _ = infer_signal_with_belief(row, df_history=df)
        baseline_signals.append(int(bs) if isinstance(bs, float) else bs)
        belief_signals.append(bls)

    df = df.copy()
    df["signal_baseline"] = baseline_signals
    df["signal_belief"]   = belief_signals

    def accuracy(signal_col, horizon):
        fwd_col = f"future_return_{horizon}d"
        if fwd_col not in df.columns:
            return np.nan
        mask = df[signal_col] != 0
        if not mask.any():
            return np.nan
        sub = df[mask]
        correct = (
            ((sub[signal_col] == 1)  & (sub[fwd_col] > 0)) |
            ((sub[signal_col] == -1) & (sub[fwd_col] < 0))
        )
        return correct.mean()

    rows = []
    for h in horizons:
        rows.append({
            "horizon":          f"{h}d",
            "n_baseline_signals": int((df["signal_baseline"] != 0).sum()),
            "n_belief_signals":   int((df["signal_belief"] != 0).sum()),
            "accuracy_baseline":  round(accuracy("signal_baseline", h) or 0, 4),
            "accuracy_belief":    round(accuracy("signal_belief",   h) or 0, 4),
        })

    result = pd.DataFrame(rows)
    if not result.empty and "accuracy_baseline" in result.columns:
        result["delta_accuracy"] = result["accuracy_belief"] - result["accuracy_baseline"]

    return result


if __name__ == "__main__":
    # Quick self-test
    import sys
    print("Testing infer_signal_with_belief...")

    # Row with both price and belief data
    row_both = pd.Series({
        "match_score": 0.7, "lead_lag_minutes": 30.0,
        "t0_volatility_20d": 0.02, "t0_trend_50_200": 0.01,
        "t0_momentum_20d": 0.03, "t0_drawdown": -0.05,
        "polymarket_prob": 0.72, "b_change": 0.04,
        "b_momentum": 0.68, "b_volatility": 0.02, "b_acceleration": 0.01,
    })
    sig, reason = infer_signal_with_belief(row_both)
    print(f"  Both present: signal={sig}, reason={reason}")
    assert sig == 1, f"Expected 1, got {sig}"

    # Row with belief only (price void)
    row_belief = pd.Series({
        "match_score": float("nan"), "lead_lag_minutes": float("nan"),
        "t0_volatility_20d": 0.03, "t0_trend_50_200": 0.005,
        "t0_momentum_20d": 0.02,
        "polymarket_prob": 0.78, "b_change": 0.06,
        "b_momentum": 0.74, "b_volatility": 0.02, "b_acceleration": 0.02,
    })
    sig2, reason2 = infer_signal_with_belief(row_belief)
    print(f"  Belief only:  signal={sig2}, reason={reason2}")
    assert sig2 == 1, f"Expected 1, got {sig2}"

    # Row bearish
    row_bear = pd.Series({
        "match_score": 0.65, "lead_lag_minutes": -20.0,
        "t0_volatility_20d": 0.05, "t0_trend_50_200": -0.01,
        "t0_momentum_20d": -0.02,
        "polymarket_prob": 0.28, "b_change": -0.05,
        "b_momentum": 0.31, "b_volatility": 0.03, "b_acceleration": -0.01,
    })
    sig3, reason3 = infer_signal_with_belief(row_bear)
    print(f"  Bearish:      signal={sig3}, reason={reason3}")
    assert sig3 == -1, f"Expected -1, got {sig3}"

    print("✓ All tests pass")
