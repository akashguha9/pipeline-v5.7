# S9 Build Architecture — May 13 2026
# Source: Systems architecture mapping session, April 2026
# This is the implementation target for the S9 build phase.
#
# LAYER MAP:
#
# core/
#   scoring.py      CE composite edge: SBE × ASS × PSE × MTL × NSD × RTT × ZPD
#                   + BCP balance_coefficient (σ/μ across 7 components)
#                   + SBC survivor_penalty (sqrt(f) for NARROW patterns)
#                   + SSD signal_discount (1 - SSD × 0.40)
#   decision.py     IL action selection: FIRE_MODE / STAGGER / PROBE / WATCHLIST
#                   ATT state gate: ACTIVE / NEAR_ACTIVATION / DORMANT_VALID
#                   LALO gate: OPEN / LATE / SATURATION / LOCKED_OUT
#   risk.py         EQS instrument_fidelity + spread + slippage
#                   OSM pre-gate: arousal + pressure + clutter + identity + realm
#                   ATOM field-count enforcer: 35-field limit
#   explain.py      BCP cv + balance_coefficient display
#                   SSD 5-component breakdown
#                   NDI IL + AV + BI + SI breakdown
#                   ATT prerequisite chain display
#
# adapters/
#   kalshi.py       Kalshi REST + WebSocket → normalized event schema
#   polymarket.py   Polymarket Gamma API → normalized event schema
#   reddit.py       PRAW → normalized event schema
#   rss.py          feedparser → normalized event schema
#   news.py         newspaper3k → normalized event schema
#   twitter.py      Grok API (xAI credit) → normalized event schema
#
# retrieval/
#   keywords.py     KeyBERT topic extraction
#   sentiment.py    FinBERT + HuggingFace models
#   fuzzy.py        rapidfuzz entity matching
#   scheduler.py    APScheduler poll loop
#
# moltbook/         Trade ledger JSON (already started)
#   *.json          Per-trade attribution records
#   mw_direction_v1_2026_04_19.json
#
# research/         Experimental mechanisms (not yet promoted to ATOM)
#   new mechanism prototypes
#   backtest scripts
#   calibration notebooks
#
# governance/
#   settings.py     All threshold constants (ATT_LAMBDA, SSD_MAX_DISCOUNT,
#                   SBC_ROBUST_THRESHOLD, BCP floor, LALO thresholds, etc.)
#   version_registry.json
#   experiment_log.json
#
# NORMALIZED EVENT SCHEMA (all adapters output this):
# {
#   "source": "kalshi | polymarket | reddit | rss | news",
#   "timestamp": "ISO8601",
#   "entity": "ticker or topic",
#   "signal_type": "prediction_market | sentiment | news | social",
#   "raw_payload": {},
#   "normalized_payload": {},
#   "confidence": 0.00
# }
#
# PROMOTION GATE (research/ → core/):
# P_m = 1 if: accuracy > alpha AND repeatability > beta AND explainability > gamma
# Default gates: alpha=0.60, beta=0.70, gamma=0.50
# All three must pass simultaneously (same logic as ZPD CLEAN_ZONE)
