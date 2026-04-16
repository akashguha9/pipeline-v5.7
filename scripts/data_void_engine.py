"""
data_void_engine.py
====================
Fallback Inference Stack for CAL DATA_VOID condition.
Integrates with signal_engine.py's infer_signal() as a drop-in upgrade.

When D_t = ∅ OR |D_t| < k:
  Signal_t = α·Historical + β·Proxy + γ·Narrative + δ·Prior
  α + β + γ + δ = 1

Pipeline position: S0 SENSE layer → fires before signal_engine.py returns "no_signal"
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional


# ── STRUCTURAL PRIORS ──────────────────────────────────────────────────────────
# Layer 5: known economic causality chains
# Source: pipeline S7B MACRO_TRANSMISSION_ENGINE
STRUCTURAL_PRIORS: dict[str, dict] = {
    # macro → asset direction
    "oil_up":        {"inflation": +1, "equities": -1, "bonds": -1, "gold": +1},
    "oil_down":      {"inflation": -1, "equities": +1, "bonds": +1, "gold": -1},
    "war_escalation":{"risk_off": +1, "bonds": +1, "gold": +1, "equities": -1},
    "war_deescalation":{"risk_off":-1,"bonds": -1, "gold": -1, "equities": +1},
    "fed_hike":      {"bonds": -1, "equities": -1, "usd": +1, "gold": -1},
    "fed_cut":       {"bonds": +1, "equities": +1, "usd": -1, "gold": +1},
    "recession_fear":{"equities": -1, "bonds": +1, "gold": +1, "usd": +1},
    "inflation_high":{"bonds": -1, "equities": -1, "gold": +1, "commodities": +1},
    "election_risk": {"volatility": +1, "equities": -1, "bonds": +0.5},
    "sanctions":     {"target_equities": -1, "commodities": +1, "risk_off": +1},
    "default_risk":  {"bonds": -1, "equities": -1, "cds": +1},
    "supply_shock":  {"inflation": +1, "equities": -1, "commodities": +1},
}

# Proxy correlation map: if primary asset is missing, use these
PROXY_MAP: dict[str, list[str]] = {
    "SPY":   ["QQQ", "IVV", "VTI", "ES=F"],
    "QQQ":   ["SPY", "XLK", "SOXX"],
    "GLD":   ["IAU", "GC=F", "SLV"],
    "USO":   ["BNO", "CL=F", "XLE"],
    "TLT":   ["IEF", "BND", "ZB=F"],
    "EEM":   ["VWO", "IEMG", "EWZ"],
    "VXX":   ["UVXY", "VIXY", "^VIX"],
    "EWZ":   ["EEM", "BRAZIL_ETF"],
    "UNG":   ["FCG", "XLE"],
    "BTC":   ["ETH", "GBTC", "BITO"],
    "DXY":   ["UUP", "EURUSD"],
}

# Regime-based directional priors
REGIME_PRIORS: dict[str, int] = {
    "DISINFLATIONARY_RISK_ON":  +1,
    "INFLATION_SQUEEZE":        -1,
    "GROWTH_SCARE":             -1,
    "LIQUIDITY_FLOOD":          +1,
    "CREDIT_CRUNCH":            -1,
    "GEOPOLITICAL_FRAGMENTATION": 0,
    "TREND":                    +1,
    "MEAN_REVERSION":            0,
    "SHOCK":                    -1,
    "MANIPULATED":               0,
    "DEAD_ZONE":                 0,
    "KALI_YUGA":                 0,
}


# ── DATA VOID DETECTION ────────────────────────────────────────────────────────

def detect_data_void(row: pd.Series, min_score: float = 0.3) -> tuple[bool, list[str]]:
    """
    Returns (is_void, reasons).
    DATA VOID when:
      - score is NaN or below threshold
      - lag is NaN
      - missing key price columns
    """
    reasons = []
    score = row.get("match_score")
    lag   = row.get("lead_lag_minutes")

    if pd.isna(score):
        reasons.append("match_score_missing")
    elif score < min_score:
        reasons.append(f"match_score_low ({score:.3f}<{min_score})")

    if pd.isna(lag):
        reasons.append("lead_lag_missing")

    # Check forward return availability
    fwd_cols = [c for c in row.index if c.startswith("future_return_")]
    if fwd_cols and all(pd.isna(row.get(c)) for c in fwd_cols):
        reasons.append("future_returns_all_missing")

    return bool(reasons), reasons


# ── LAYER 1: HISTORICAL ANALOGUE ──────────────────────────────────────────────

def layer1_historical(row: pd.Series, df_history: Optional[pd.DataFrame]) -> tuple[float, float]:
    """
    Search historical rows for analogues on:
      - same seed_label
      - similar regime (t0_trend_50_200 sign)
      - similar volatility regime (vol quartile)

    Returns (direction_score, confidence) where direction_score ∈ [-1,+1].
    """
    if df_history is None or df_history.empty:
        return 0.0, 0.0

    seed = row.get("seed_label", "")
    vol  = row.get("t0_volatility_20d", np.nan)
    trend = row.get("t0_trend_50_200", np.nan)

    analogues = df_history.copy()

    # Filter: same seed
    if seed and "seed_label" in analogues.columns:
        analogues = analogues[analogues["seed_label"] == seed]

    # Filter: similar trend regime
    if pd.notna(trend) and "t0_trend_50_200" in analogues.columns:
        same_sign = analogues["t0_trend_50_200"].apply(
            lambda x: np.sign(x) == np.sign(trend) if pd.notna(x) else False
        )
        analogues = analogues[same_sign]

    # Filter: similar volatility quartile
    if pd.notna(vol) and "t0_volatility_20d" in analogues.columns:
        q25 = analogues["t0_volatility_20d"].quantile(0.25)
        q75 = analogues["t0_volatility_20d"].quantile(0.75)
        if vol < q25:
            analogues = analogues[analogues["t0_volatility_20d"] < q25]
        elif vol > q75:
            analogues = analogues[analogues["t0_volatility_20d"] > q75]

    # Need at least 3 analogues
    if len(analogues) < 3:
        return 0.0, 0.0

    # Use 1d forward return as proxy for direction
    fwd_col = "future_return_1d"
    if fwd_col not in analogues.columns:
        fwd_cols = sorted([c for c in analogues.columns if c.startswith("future_return_")])
        if not fwd_cols:
            return 0.0, 0.0
        fwd_col = fwd_cols[0]

    returns = analogues[fwd_col].dropna()
    if len(returns) < 3:
        return 0.0, 0.0

    direction = np.sign(returns.mean())
    # Confidence: proportion of analogues pointing same direction
    confidence = (np.sign(returns) == direction).mean()
    return float(direction), float(confidence * 0.6)  # max 0.6 confidence from history


# ── LAYER 2: CROSS-ASSET PROXY ────────────────────────────────────────────────

def layer2_proxy(row: pd.Series) -> tuple[float, float]:
    """
    Uses momentum and trend signals from related assets as proxy.
    Since we don't have live proxy prices here, we infer from available
    momentum/trend fields in the row.

    Returns (direction_score, confidence).
    """
    signals = []

    # Existing momentum signals
    mom20 = row.get("t0_momentum_20d")
    mom60 = row.get("t0_momentum_60d")
    trend = row.get("t0_trend_50_200")

    if pd.notna(mom20):
        signals.append(np.sign(mom20))
    if pd.notna(mom60):
        signals.append(np.sign(mom60))
    if pd.notna(trend):
        signals.append(np.sign(trend))

    if not signals:
        return 0.0, 0.0

    direction = np.sign(np.mean(signals))
    agreement = (np.array(signals) == direction).mean()
    confidence = agreement * 0.5  # max 0.5 from proxy
    return float(direction), float(confidence)


# ── LAYER 3: NARRATIVE CONTINUATION ──────────────────────────────────────────

def layer3_narrative(row: pd.Series) -> tuple[float, float]:
    """
    Narrative inertia: extend the current trajectory.
    N_{t+1} = N_t + Δsentiment
    Uses: drawdown, momentum signs as narrative proxies.

    Returns (direction_score, confidence).
    """
    drawdown = row.get("t0_drawdown")
    mom20 = row.get("t0_momentum_20d")
    trend = row.get("t0_trend_50_200")

    narrative_score = 0.0
    count = 0

    if pd.notna(drawdown):
        # deep drawdown → negative narrative inertia
        if drawdown < -0.15:
            narrative_score -= 1
        elif drawdown > -0.05:
            narrative_score += 0.5
        count += 1

    if pd.notna(mom20):
        narrative_score += np.sign(mom20) * 0.7
        count += 1

    if pd.notna(trend):
        narrative_score += np.sign(trend) * 0.5
        count += 1

    if count == 0:
        return 0.0, 0.0

    direction = np.sign(narrative_score / count)
    confidence = min(0.4, abs(narrative_score / count) * 0.4)
    return float(direction), float(confidence)


# ── LAYER 4: PROBABILISTIC SIMULATION ────────────────────────────────────────

def layer4_simulation(row: pd.Series, n_scenarios: int = 500) -> tuple[float, float]:
    """
    Monte Carlo over plausible return distributions given:
      - current volatility
      - momentum regime
      - trend direction

    Returns (direction_score, confidence).
    """
    vol   = row.get("t0_volatility_20d", 0.02)
    mom20 = row.get("t0_momentum_20d", 0.0)
    trend = row.get("t0_trend_50_200", 0.0)

    if pd.isna(vol):
        vol = 0.02
    if pd.isna(mom20):
        mom20 = 0.0
    if pd.isna(trend):
        trend = 0.0

    # Drift = weighted blend of momentum signals
    drift = (
        np.sign(mom20) * 0.002 * abs(mom20) / max(abs(mom20), 0.001)
        + np.sign(trend) * 0.001
    )

    np.random.seed(42)
    simulated_returns = np.random.normal(drift, vol, n_scenarios)
    positive_fraction = (simulated_returns > 0).mean()

    if positive_fraction > 0.55:
        direction = 1.0
        confidence = (positive_fraction - 0.5) * 2 * 0.5  # scale to max 0.5
    elif positive_fraction < 0.45:
        direction = -1.0
        confidence = (0.5 - positive_fraction) * 2 * 0.5
    else:
        direction = 0.0
        confidence = 0.0

    return float(direction), float(confidence)


# ── LAYER 5: STRUCTURAL PRIORS ────────────────────────────────────────────────

def layer5_prior(row: pd.Series) -> tuple[float, float]:
    """
    Apply known macro causality chains.
    Uses seed_label and reason fields to detect applicable priors.

    Returns (direction_score, confidence).
    """
    seed  = str(row.get("seed_label", "")).lower()
    label = str(row.get("reason", "")).lower()

    matched_priors = []

    for trigger, effects in STRUCTURAL_PRIORS.items():
        if trigger.replace("_", " ") in seed or trigger.replace("_", " ") in label:
            # Take the dominant effect direction
            directions = [v for k, v in effects.items() if k not in ["inflation", "volatility", "risk_off"]]
            if directions:
                matched_priors.append(np.sign(np.mean(directions)))

    if not matched_priors:
        # Fall back to regime prior
        regime = str(row.get("regime", "")).upper()
        for regime_key, direction in REGIME_PRIORS.items():
            if regime_key in regime:
                return float(direction), 0.30
        return 0.0, 0.0

    direction = np.sign(np.mean(matched_priors))
    confidence = min(0.45, len(matched_priors) * 0.15)
    return float(direction), float(confidence)


# ── WEIGHT ALLOCATION ─────────────────────────────────────────────────────────

def allocate_weights(
    l1_conf: float,
    l2_conf: float,
    l3_conf: float,
    l4_conf: float,
    l5_conf: float,
) -> tuple[float, float, float, float, float]:
    """
    Adaptive weight allocation.
    Signal_t = α·L1 + β·L2 + γ·L3 + δ·L4 + ε·L5
    Weights proportional to layer confidence. Sum = 1.
    """
    raw = np.array([l1_conf, l2_conf, l3_conf, l4_conf, l5_conf])
    total = raw.sum()
    if total == 0:
        # Equal weights
        return 0.2, 0.2, 0.2, 0.2, 0.2
    weights = raw / total
    return tuple(weights)


# ── MASTER INFERENCE FUNCTION ─────────────────────────────────────────────────

def infer_from_void(
    row: pd.Series,
    df_history: Optional[pd.DataFrame] = None,
    min_confidence_threshold: float = 0.15,
) -> dict:
    """
    Master fallback inference. Called when data void is detected.

    Returns structured output:
    {
        "data_void_detected": True,
        "fallback_layers_used": [...],
        "proxy_signals_used": [...],
        "confidence_score": float [0-1],
        "estimated_direction": "UP" | "DOWN" | "NEUTRAL",
        "signal": int (1, -1, 0),
        "reason": str,
        "alpha": float, "beta": float, "gamma": float, "delta": float, "epsilon": float
    }
    """
    # Layer 1: Historical
    l1_dir, l1_conf = layer1_historical(row, df_history)
    # Layer 2: Proxy
    l2_dir, l2_conf = layer2_proxy(row)
    # Layer 3: Narrative
    l3_dir, l3_conf = layer3_narrative(row)
    # Layer 4: Simulation
    l4_dir, l4_conf = layer4_simulation(row)
    # Layer 5: Structural prior
    l5_dir, l5_conf = layer5_prior(row)

    # Adaptive weights
    α, β, γ, δ, ε = allocate_weights(l1_conf, l2_conf, l3_conf, l4_conf, l5_conf)

    # Composite direction score
    composite = (
        α * l1_dir +
        β * l2_dir +
        γ * l3_dir +
        δ * l4_dir +
        ε * l5_dir
    )

    # Overall confidence (weighted average of layer confidences)
    overall_confidence = (
        α * l1_conf +
        β * l2_conf +
        γ * l3_conf +
        δ * l4_conf +
        ε * l5_conf
    )

    # Layers actually used
    layers_used = []
    proxies_used = []
    if l1_conf > 0: layers_used.append("L1_historical")
    if l2_conf > 0:
        layers_used.append("L2_proxy")
        seed = row.get("seed_label", "")
        if seed in PROXY_MAP:
            proxies_used.extend(PROXY_MAP[seed][:2])
        proxies_used.extend(["t0_momentum_20d", "t0_trend_50_200"])
    if l3_conf > 0: layers_used.append("L3_narrative")
    if l4_conf > 0: layers_used.append("L4_simulation")
    if l5_conf > 0: layers_used.append("L5_prior")

    # Direction decision
    if overall_confidence < min_confidence_threshold or composite == 0:
        signal = 0
        direction_str = "NEUTRAL"
        reason = "void_inferred_neutral_low_conf"
    elif composite > 0.1:
        signal = 1
        direction_str = "UP"
        reason = "void_inferred_up"
    elif composite < -0.1:
        signal = -1
        direction_str = "DOWN"
        reason = "void_inferred_down"
    else:
        signal = 0
        direction_str = "NEUTRAL"
        reason = "void_inferred_neutral_weak_signal"

    # Enrich reason string with layer context
    reason_parts = [reason]
    if l2_conf > 0:
        trend = row.get("t0_trend_50_200")
        if pd.notna(trend):
            reason_parts.append("uptrend" if trend > 0 else "downtrend")
    vol = row.get("t0_volatility_20d")
    if pd.notna(vol) and vol > 0.04:
        reason_parts.append("high_vol")
    reason = "_".join(reason_parts)

    return {
        "data_void_detected": True,
        "fallback_layers_used": layers_used,
        "proxy_signals_used": proxies_used,
        "confidence_score": round(float(overall_confidence), 4),
        "estimated_direction": direction_str,
        "signal": signal,
        "reason": reason,
        "composite_score": round(float(composite), 4),
        # Layer weights (α+β+γ+δ+ε=1)
        "alpha": round(float(α), 3),   # historical weight
        "beta":  round(float(β), 3),   # proxy weight
        "gamma": round(float(γ), 3),   # narrative weight
        "delta": round(float(δ), 3),   # simulation weight
        "epsilon": round(float(ε), 3), # prior weight
        # Per-layer details
        "l1_dir": l1_dir, "l1_conf": l1_conf,
        "l2_dir": l2_dir, "l2_conf": l2_conf,
        "l3_dir": l3_dir, "l3_conf": l3_conf,
        "l4_dir": l4_dir, "l4_conf": l4_conf,
        "l5_dir": l5_dir, "l5_conf": l5_conf,
    }


# ── UPGRADED infer_signal FUNCTION ───────────────────────────────────────────

def infer_signal_with_void_fallback(
    row: pd.Series,
    df_history: Optional[pd.DataFrame] = None,
    min_score: float = 0.3,
) -> tuple[int, str]:
    """
    Drop-in replacement for signal_engine.infer_signal().
    
    USAGE in signal_engine.py:
        from data_void_engine import infer_signal_with_void_fallback as infer_signal
    
    Returns (signal: int, reason: str) — same interface as original.
    Signal: 1 = UP, -1 = DOWN, 0 = NEUTRAL
    """
    is_void, void_reasons = detect_data_void(row, min_score)

    if not is_void:
        # Standard path — original logic preserved exactly
        score = row.get("match_score")
        lag   = row.get("lead_lag_minutes")
        vol   = row.get("t0_volatility_20d")
        trend = row.get("t0_trend_50_200")
        mom20 = row.get("t0_momentum_20d")

        signal = 0
        reason = "no_signal"

        if score >= min_score and lag > 0:
            signal = 1
            reason = "news_led"
        elif score >= min_score and lag < 0:
            signal = -1
            reason = "market_led"

        if pd.notna(vol) and vol > 0.04:
            reason += "_high_vol"
        if pd.notna(trend):
            reason += "_uptrend" if trend > 0 else "_downtrend"
        if pd.notna(mom20):
            reason += "_mom_up" if mom20 > 0 else "_mom_down"

        return signal, reason

    # DATA VOID PATH — fallback inference
    result = infer_from_void(row, df_history)
    return result["signal"], result["reason"]


# ── REPORTING UTILITY ─────────────────────────────────────────────────────────

def format_void_report(result: dict) -> str:
    """Print-friendly void inference report."""
    lines = [
        "━━ DATA VOID INFERENCE REPORT ━━",
        f"  Data Void Detected:    {result['data_void_detected']}",
        f"  Fallback Layers Used:  {', '.join(result['fallback_layers_used']) or 'none'}",
        f"  Proxy Signals Used:    {', '.join(result['proxy_signals_used']) or 'none'}",
        f"  Confidence Score:      {result['confidence_score']:.4f}",
        f"  Estimated Direction:   {result['estimated_direction']}",
        f"  Composite Score:       {result['composite_score']:+.4f}",
        f"  Weights (α,β,γ,δ,ε):  "
        f"{result['alpha']:.2f}, {result['beta']:.2f}, {result['gamma']:.2f}, "
        f"{result['delta']:.2f}, {result['epsilon']:.2f}",
        f"  Signal:                {result['signal']} ({result['reason']})",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running data_void_engine self-test...\n")

    # Test 1: Complete data void (NaN score + NaN lag)
    row_void = pd.Series({
        "match_score": np.nan,
        "lead_lag_minutes": np.nan,
        "t0_volatility_20d": 0.025,
        "t0_momentum_20d": 0.03,
        "t0_momentum_60d": 0.05,
        "t0_trend_50_200": 0.01,
        "t0_drawdown": -0.08,
        "seed_label": "macro_spy",
    })

    is_void, reasons = detect_data_void(row_void)
    print(f"Test 1 — Complete void: {is_void} | Reasons: {reasons}")
    result = infer_from_void(row_void)
    print(format_void_report(result))
    signal, reason = infer_signal_with_void_fallback(row_void)
    print(f"  → infer_signal_with_void_fallback: signal={signal}, reason={reason}\n")

    # Test 2: Low score (below threshold)
    row_low = pd.Series({
        "match_score": 0.1,
        "lead_lag_minutes": 15.0,
        "t0_volatility_20d": 0.045,
        "t0_momentum_20d": -0.02,
        "t0_momentum_60d": -0.04,
        "t0_trend_50_200": -0.005,
        "t0_drawdown": -0.20,
        "seed_label": "oil_geopolitics",
        "reason": "war_escalation",
    })

    is_void, reasons = detect_data_void(row_low)
    print(f"Test 2 — Low score void: {is_void} | Reasons: {reasons}")
    result2 = infer_from_void(row_low)
    print(format_void_report(result2))
    signal2, reason2 = infer_signal_with_void_fallback(row_low)
    print(f"  → infer_signal_with_void_fallback: signal={signal2}, reason={reason2}\n")

    # Test 3: Good data — should use original path
    row_good = pd.Series({
        "match_score": 0.75,
        "lead_lag_minutes": 30.0,
        "t0_volatility_20d": 0.018,
        "t0_momentum_20d": 0.04,
        "t0_momentum_60d": 0.06,
        "t0_trend_50_200": 0.002,
        "t0_drawdown": -0.03,
        "seed_label": "macro_spy",
    })

    is_void3, _ = detect_data_void(row_good)
    signal3, reason3 = infer_signal_with_void_fallback(row_good)
    print(f"Test 3 — Good data (no void): is_void={is_void3}")
    print(f"  → signal={signal3}, reason={reason3}")
    print("\n✓ Self-test complete.")
