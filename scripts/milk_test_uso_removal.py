"""
milk_test_uso_removal.py
========================
Run from repo root after applying USO removal.
Validates the change without touching the actual pipeline.
"""
import sys
from pathlib import Path

PASS = 0
FAIL = 0

def check(cond, msg_pass, msg_fail):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {msg_pass}")
        PASS += 1
    else:
        print(f"  FAIL  {msg_fail}")
        FAIL += 1

print("\n=== MILK TEST: USO removal ===\n")

# 1. Verify USO is gone from all 4 src/ files
files = [
    "src/backtest_filtered_overlay.py",
    "src/backtest_filtered_overlay_costs.py",
    "src/walkforward_filtered_overlay.py",
    "src/export_latest_filtered_signals.py",
]
for path in files:
    p = Path(path)
    if not p.exists():
        check(False, "", f"{path} not found")
        continue
    text = p.read_text()
    # USO must not appear in symbol context
    # Allow 'USO' in comments but not in string literals
    import re
    uso_in_strings = re.findall(r'''['"]USO['"]''', text)
    check(
        len(uso_in_strings) == 0,
        f"{path}: USO not present in string literals",
        f"{path}: USO still found {uso_in_strings} — change may not have applied"
    )
    # GLD and GDAXI must still be present
    check("GLD" in text, f"{path}: GLD still present", f"{path}: GLD missing!")
    check("GDAXI" in text or "^GDAXI" in text, f"{path}: GDAXI still present", f"{path}: GDAXI missing!")

# 2. Output CSV check (if available)
import pandas as pd

summary_path = Path("data/processed/paper_trade_filtered_overlay_summary.csv")
if summary_path.exists():
    df = pd.read_csv(summary_path)
    print(f"\n  Summary CSV found. Contents:")
    print(df.to_string(index=False))
    
    # n_trades should be ~171 (was 235, USO had 64)
    if 'n_trades' in df.columns:
        n = int(df['n_trades'].iloc[0])
        check(
            140 <= n <= 200,
            f"n_trades={n} is in expected range [140, 200]",
            f"n_trades={n} is outside expected range — USO removal may not have run"
        )
        check(
            n < 235,
            f"n_trades={n} < 235 (baseline) — trades were correctly filtered",
            f"n_trades={n} >= 235 — no reduction, check if pipeline was re-run"
        )
else:
    print("\n  Summary CSV not found — run pipeline first, then re-run milk test")

# 3. Synthetic filter test — no dependencies on actual data
print("\n  Synthetic filter check:")
import io
test_csv = """Ticker,signal,overlay_score,future_ret_5d
GLD,1,0.62,0.015
USO,1,0.63,0.010
^GDAXI,1,0.68,0.005
SPY,1,0.65,0.008
GLD,1,0.58,-0.003
USO,1,0.70,-0.005
"""
test_df = pd.read_csv(io.StringIO(test_csv))
champion = test_df[
    (test_df['signal'] == 1) &
    (test_df['Ticker'].isin({"GLD", "^GDAXI"})) &  # no USO
    (test_df['overlay_score'].abs() >= 0.55) &
    (test_df['overlay_score'].abs() < 0.75)
]
check(len(champion) == 3, f"Filter returns 3 trades (GLD×2 + GDAXI×1)", f"Filter returned {len(champion)}")
check("USO" not in champion['Ticker'].values, "USO correctly excluded", "USO still in results")
check("GLD" in champion['Ticker'].values, "GLD included", "GLD missing")
check("^GDAXI" in champion['Ticker'].values, "^GDAXI included", "^GDAXI missing")
check("SPY" not in champion['Ticker'].values, "SPY correctly excluded", "SPY in results")

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL PASS")
    sys.exit(0)
else:
    print("FAILURES — review output above")
    sys.exit(1)
