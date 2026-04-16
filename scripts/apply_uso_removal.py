"""
apply_uso_removal.py
====================
Run from repo root. Applies the single refinement: remove USO from symbol gate.

Changes exactly one thing in each of the four src/ files:
  'USO' is removed from the CHAMPION_SYMBOLS / symbol filter set.

The script:
1. Reads each file
2. Shows every line containing 'USO' (dry-run first)
3. Replaces the symbol set string — three patterns cover all likely forms
4. Writes back only if a change was made
5. Prints a diff summary
"""

import re
import sys
from pathlib import Path

FILES = [
    "src/backtest_filtered_overlay.py",
    "src/backtest_filtered_overlay_costs.py",
    "src/walkforward_filtered_overlay.py",
    "src/export_latest_filtered_signals.py",
]

# Replacement patterns — covers every reasonable way the set can be written.
# Each pattern is (regex_to_match, replacement_string).
# We try all three; whichever matches first wins.
PATTERNS = [
    # Pattern A: set literal with USO in the middle or at end
    # e.g. {"GLD", "^GDAXI", "USO"} or {'GLD', '^GDAXI', 'USO'}
    (r'''\{['"](GLD|GLD)['"]\s*,\s*['"](\^GDAXI|USO)['"]\s*,\s*['"](USO|\^GDAXI)['"]\}''',
     '{"GLD", "^GDAXI"}'),

    # Pattern B: list literal
    # e.g. ["GLD", "^GDAXI", "USO"] or ["GLD", "USO", "^GDAXI"]
    (r'''\[['"](GLD)['"]\s*,\s*['"](\^GDAXI|USO)['"]\s*,\s*['"](\^GDAXI|USO)['"]\]''',
     '["GLD", "^GDAXI"]'),

    # Pattern C: individual .isin() or similar with USO as one element
    # e.g. isin(["GLD", "^GDAXI", "USO"])
    (r'''isin\(\s*\[[^\]]*['"]USO['"]\s*,?\s*[^\]]*\]\s*\)''', None),  # handled below

    # Pattern D: CHAMPION_SYMBOLS = ... style constant
    # e.g. CHAMPION_SYMBOLS = {"GLD", "^GDAXI", "USO"}
    (r'''CHAMPION_SYMBOLS\s*=\s*\{[^}]*['"]USO['"][^}]*\}''', None),  # handled below
]

def remove_uso_from_set_string(s: str) -> str:
    """Remove 'USO' from any set/list/tuple literal in string s."""
    # Handle all quote styles and orderings
    # Removes:  , "USO"  or  "USO",  or  , 'USO'  or  'USO',
    s = re.sub(r''',\s*['"]USO['"]\s*''', '', s)
    s = re.sub(r'''\s*['"]USO['"]\s*,\s*''', '', s)
    s = re.sub(r'''\s*['"]USO['"]\s*''', '', s)  # if USO is the only element (rare)
    return s


def process_file(path: str, dry_run: bool = True) -> tuple[bool, str, str]:
    """
    Returns (changed: bool, original: str, modified: str).
    """
    p = Path(path)
    if not p.exists():
        print(f"  SKIP (not found): {path}")
        return False, "", ""

    original = p.read_text(encoding="utf-8")
    modified = original

    # Apply the USO removal to every line that contains USO in a symbol context
    lines_in  = original.splitlines(keepends=True)
    lines_out = []
    changed_lines = []

    for i, line in enumerate(lines_in, 1):
        # Only modify lines that contain USO in what looks like a symbol list/set
        if "USO" in line and any(kw in line for kw in [
            '"GLD"', "'GLD'", "isin", "SYMBOL", "symbol", "champion", "filter",
            "CHAMPION", "{", "[", "in ["
        ]):
            new_line = remove_uso_from_set_string(line)
            if new_line != line:
                changed_lines.append((i, line.rstrip(), new_line.rstrip()))
                lines_out.append(new_line)
            else:
                lines_out.append(line)
        else:
            lines_out.append(line)

    modified = "".join(lines_out)

    if modified == original:
        return False, original, original

    return True, original, modified


def main(dry_run: bool = True):
    mode = "DRY RUN" if dry_run else "APPLYING CHANGES"
    print(f"\n{'='*60}")
    print(f"USO REMOVAL — {mode}")
    print(f"{'='*60}\n")

    any_change = False
    for path in FILES:
        print(f"--- {path} ---")
        changed, original, modified = process_file(path, dry_run)

        if not changed:
            # Show USO lines even if no automated change found — manual review needed
            p = Path(path)
            if p.exists():
                for i, line in enumerate(p.read_text().splitlines(), 1):
                    if "USO" in line:
                        print(f"  Line {i:4d}: {line.rstrip()}")
                        print(f"         → MANUAL REVIEW: does this contain the symbol set?")
            print(f"  No automated change made. Review lines above.\n")
            continue

        any_change = True
        # Show what changed
        orig_lines = original.splitlines(keepends=True)
        mod_lines  = modified.splitlines(keepends=True)
        for i, (o, m) in enumerate(zip(orig_lines, mod_lines), 1):
            if o != m:
                print(f"  Line {i:4d} BEFORE: {o.rstrip()}")
                print(f"  Line {i:4d} AFTER:  {m.rstrip()}")

        if not dry_run:
            Path(path).write_text(modified, encoding="utf-8")
            print(f"  ✓ Written.")
        else:
            print(f"  [DRY RUN — not written]")
        print()

    return any_change


if __name__ == "__main__":
    dry = "--apply" not in sys.argv
    if dry:
        print("Running in DRY RUN mode. Pass --apply to write changes.")
    else:
        print("APPLYING changes to files.")

    changed = main(dry_run=dry)

    if dry and changed:
        print("\nTo apply: python apply_uso_removal.py --apply")
    elif dry and not changed:
        print("\nNo automated changes found.")
        print("Paste the content of your src/ files here for manual diff.")
