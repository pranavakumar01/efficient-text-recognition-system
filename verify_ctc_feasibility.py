"""
Verify CTC feasibility through the REAL preprocessing pipeline.

This runs src.preprocessing.enhancement.ImagePreprocessor.prepare() - the exact
function src/dataset.py calls - over every image the default training config
would load, then replicates the width arithmetic from ocr_collate_fn and checks
the condition CTC actually requires:

    T  >=  L + (number of adjacent duplicate characters in the label)

The duplicate term matters and is easy to miss: CTC must emit a blank between
two identical consecutive characters, so "seen" needs 5 timesteps, not 4. A
sample that fails this has no valid alignment, its loss is +inf, and
zero_infinity=True silently replaces it with zero - contributing no gradient
while still being averaged into the reported loss.

Runs without torch, so it can be checked in the sandbox before the user spends
hours on a real run.
"""
import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.enhancement import ImagePreprocessor

WIDTH_REDUCTION = 4
MANIFEST = os.path.join("data", "splits.csv")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def min_timesteps_required(encoded):
    """L plus one extra slot for each adjacent duplicate."""
    if not encoded:
        return 1
    need = len(encoded)
    for a, b in zip(encoded, encoded[1:]):
        if a == b:
            need += 1
    return need


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    root = os.path.dirname(os.path.abspath(MANIFEST))

    def select(split):
        return [r for r in rows
                if r["split"] == split and r["is_math"] == "0" and r["ctc_feasible"] == "1"]

    # Vocabulary exactly as train.py builds it: from the training labels in use.
    train_rows = select("train")
    vocab = "".join(sorted({c for r in train_rows for c in r["label"]}))
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    print(f"vocab: {len(vocab)} chars -> {len(vocab) + 1} classes with blank")

    pre = ImagePreprocessor(target_height=32, default_max_width=640)
    grand_fail = 0

    for split in ("train", "valid", "test"):
        sel = select(split)
        if LIMIT:
            sel = sel[:LIMIT]
        fails, widths, ratios = [], [], []
        t0 = time.time()

        for i, r in enumerate(sel):
            path = os.path.join(root, os.path.normpath(r["image_path"].replace("\\", "/")))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                fails.append((r["label"], "UNREADABLE", 0, 0))
                continue

            _, content_width = pre.prepare(img, max_width=640)
            T = max(1, content_width // WIDTH_REDUCTION)
            encoded = [char2idx[c] for c in r["label"] if c in char2idx]
            need = min_timesteps_required(encoded)

            widths.append(content_width)
            ratios.append(T / max(1, need))
            if T < need:
                fails.append((r["label"], f"T={T} need={need}", content_width, len(encoded)))

            if (i + 1) % 250 == 0:
                print(f"  {split}: {i + 1}/{len(sel)} ({time.time() - t0:.0f}s)", flush=True)

        n = len(sel)
        ok = n - len(fails)
        print(f"\n[{split}] {n} samples in {time.time() - t0:.0f}s")
        print(f"   alignable : {ok}/{n} ({100.0 * ok / max(1, n):.2f}%)")
        if widths:
            widths_sorted = sorted(widths)
            print(f"   content_width  min/median/max : {widths_sorted[0]} / "
                  f"{widths_sorted[len(widths_sorted) // 2]} / {widths_sorted[-1]}")
            print(f"   T/need ratio   min/median      : {min(ratios):.2f} / "
                  f"{sorted(ratios)[len(ratios) // 2]:.2f}   (>=1.0 required)")
            tight = sum(1 for x in ratios if x < 1.5)
            print(f"   samples with less than 1.5x headroom : {tight}")
        if fails:
            grand_fail += len(fails)
            print(f"   [X] {len(fails)} INFEASIBLE - CTC loss will be zeroed for these:")
            for label, why, cw, L in fails[:8]:
                print(f"       {why:<18} width={cw:<4} len={L:<3} {label[:44]!r}")

    print("\n" + "=" * 68)
    if grand_fail == 0:
        print("[OK] Every sample in the default config has a valid CTC alignment,")
        print("     including the blank required between repeated characters.")
        print("     No sample will have its loss silently zeroed.")
    else:
        print(f"[X] {grand_fail} sample(s) cannot be aligned. Raise max_width or")
        print("    tighten the make_splits feasibility rule before training.")
    print("=" * 68)
    return 1 if grand_fail else 0


if __name__ == "__main__":
    sys.exit(main())
