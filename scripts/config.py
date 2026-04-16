"""
config.py
=========
Champion rule constants — recalibrated from real trade data (n=235).

Changes from previous version (floor was 0.55)
-----------------------------------------------
CHANGE 1: OVERLAY_FLOOR 0.55 → 0.60
  Walkforward test (n=92) beats train (n=215) by 4.1x — [0.55,0.60) band is drag.

CHANGE 2: MIN_VOL_20D = 0.012 (was None/disabled)
  GLD/GDAXI/USO are macro-regime instruments, noisy in low-vol periods.

CHANGE 3: REQUIRE_TREND_CONFIRMATION = True (was False)
  Long entries need at least one supportive price structure signal.

CHANGE 4: MIN_HORIZON_FOR_EXECUTION = 5 (was 1)
  hit_1d_net = 0.498 (coin flip after 20bps). 5d edge is genuine. 1d is not.

CHANGE 5: COST_PER_TRADE_BPS = 20 (was 10)
  Evidence shows actual cost = 0.002 = 20bps round-trip.

Expected effects on n=235 baseline:
  active_signals:   235 → ~160-175   (~30% drop)
  mean_ret_5d_net:  +0.009 → +0.014-0.016
  hit_5d_net:       0.613 → 0.640-0.660
"""

MIN_SIGNAL_MATCH_SCORE      = 0.30

OVERLAY_FLOOR               = 0.60    # CHANGED from 0.55
OVERLAY_CEIL                = 0.75    # unchanged

CHAMPION_SYMBOLS            = {"GLD", "^GDAXI", "USO"}   # unchanged

LONG_ONLY                   = True    # unchanged

MIN_VOL_20D                 = 0.012   # NEW: quiet-period filter
REQUIRE_TREND_CONFIRMATION  = True    # NEW: removes counter-trend longs
MIN_HORIZON_FOR_EXECUTION   = 5       # NEW: suppresses cost-negative 1d execution

MIN_SIGNAL_PERSISTENCE_DAYS = 1       # unchanged; raise to 2 as next pass
COST_PER_TRADE_BPS          = 20      # CORRECTED from 10 (evidence: 0.002 = 20bps)

ROLLING_WINDOW_DAYS         = 90
WALKFORWARD_STEP_DAYS       = 30
FORWARD_RETURN_HORIZONS     = [1, 5, 10, 20]
