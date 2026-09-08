"""
Build a single canonical train/val/test manifest for every dataset in data/.

Why this file exists
--------------------
Before this, `train.py` loaded all four data directories and did a fresh
`random_split(0.85/0.15)` on every run, while `evaluate.py` defaulted to
`data/expanded` with `split=None`. Three consequences:

  1. The evaluation set was inside the training set, so reported CER was
     measuring memorisation, not recognition.
  2. The split changed on every run, so two training runs were never
     comparable and no improvement could be detected.
  3. `data/expanded` repeats label strings across different image files
     (e.g. "sigma(z) = 1 / (1 + e^(-z))" appears at samples 0007, 0010 and
     0012). A row-level random split puts one copy in train and another in
     test, which leaks the answer even when the split is otherwise honest.

This script fixes all three by grouping rows on their *normalised label* and
assigning whole groups to a split, then writing the result to disk so every
downstream script reads the identical partition.

It also records, per row, whether the sample is even representable under CTC.
CTC needs one timestep per output character, and the CNN produces
T = floor(W_resized / 4) timesteps where W_resized = target_h * (w / h).
When T < len(label) no valid alignment exists, `zero_infinity=True` silently
zeroes the loss, and the sample trains nothing while still skewing the class
prior. Those rows are flagged so training can exclude them deliberately
instead of wasting 40% of the dataset without knowing it.

Usage
-----
    python -m src.make_splits
    python -m src.make_splits --test-frac 0.2 --seed 7
"""

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict

# Datasets to scan, relative to the data root.
DATASET_DIRS = ["expanded", "kaggle_dataset", "mathwriting", "synthetic"]

MANIFEST_COLUMNS = [
    "dataset",
    "image_path",   # relative to the data root, forward slashes
    "label",
    "split",
    "source_split", # split declared by the source CSV, blank if none
    "is_math",
    "width",
    "height",
    "timesteps",    # T = floor(target_h * (w/h) / 4)
    "label_len",    # L
    "ctc_feasible", # 1 when T >= L
]

# LaTeX / math markers. A label containing any of these cannot be laid out
# left-to-right in a single line of pixels, which is what CTC assumes.
_MATH_PATTERN = re.compile(r"\\[a-zA-Z]+|\\\\|[\^_]\{|\\frac|\\sqrt|\\int|\\sum")


def normalise_label(label: str) -> str:
    """Group key for leakage control: case- and whitespace-insensitive."""
    return re.sub(r"\s+", " ", label.strip().lower())


def looks_like_math(label: str, category: str) -> bool:
    if category and category.strip().lower() == "mathwriting":
        return True
    return bool(_MATH_PATTERN.search(label or ""))


def read_image_size(path: str):
    """Return (width, height) or (0, 0) if unreadable. Prefers PIL (no full decode)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.shape[1], img.shape[0]
    except Exception:
        pass
    return 0, 0


def stable_bucket(key: str, buckets: int = 1000) -> int:
    """
    Deterministic hash -> bucket. Uses md5 rather than the builtin hash(),
    which is randomised per interpreter run and would silently reshuffle the
    split between sessions.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def collect_rows(data_root: str, target_h: int = 32):
    """Scan every dataset directory and return a list of manifest row dicts."""
    rows = []
    missing_files = Counter()

    for dataset in DATASET_DIRS:
        ds_dir = os.path.join(data_root, dataset)
        csv_path = os.path.join(ds_dir, "annotations.csv")
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rel_raw = (row.get("image_path") or "").strip()
                label = row.get("label")
                if not rel_raw or label is None or label == "":
                    continue

                # Annotation CSVs are written on Windows and use backslashes.
                # Normalise so the manifest is portable.
                rel = rel_raw.replace("\\", "/")
                abs_path = os.path.join(ds_dir, os.path.normpath(rel))
                if not os.path.exists(abs_path):
                    missing_files[dataset] += 1
                    continue

                w, h = read_image_size(abs_path)
                if w <= 0 or h <= 0:
                    missing_files[dataset] += 1
                    continue

                resized_w = max(16, int(target_h * (w / float(h))))
                timesteps = resized_w // 4
                label_len = len(label)

                rows.append({
                    "dataset": dataset,
                    "image_path": f"{dataset}/{rel}",
                    "label": label,
                    "split": "",  # assigned later
                    "source_split": (row.get("split") or "").strip().lower(),
                    "is_math": int(looks_like_math(label, row.get("category") or "")),
                    "width": w,
                    "height": h,
                    "timesteps": timesteps,
                    "label_len": label_len,
                    "ctc_feasible": int(timesteps >= label_len),
                })

    for dataset, count in missing_files.items():
        print(f"[!] {dataset}: skipped {count} row(s) with a missing or unreadable image")

    return rows


def assign_splits(rows, val_frac: float, test_frac: float, seed: int):
    """
    Assign whole label-groups to splits so no label string appears in more
    than one split.

    A dataset that ships its own `split` column is authoritative - mathwriting
    provides train/valid/test and there is no reason to discard a partition the
    upstream authors defined. Those rows keep their given split, and that
    choice propagates to every other row sharing the same label so grouping
    still holds. Everything else is assigned by a stable hash of the label,
    stratified by (dataset, is_math) so each split keeps a similar composition
    rather than, say, all the French surnames landing in test.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[normalise_label(row["label"])].append(row)

    val_cut = int(round(val_frac * 1000))
    test_cut = val_cut + int(round(test_frac * 1000))
    valid_names = {"train", "valid", "test"}

    honoured = 0
    conflicts = []
    unassigned = []

    for key, members in groups.items():
        declared = {m["source_split"] for m in members
                    if m["source_split"] in valid_names}
        if not declared:
            unassigned.append(key)
            continue
        if len(declared) > 1:
            # Same label declared into two different upstream splits. Send the
            # whole group to the strictest destination so nothing leaks into
            # training, and report it.
            conflicts.append((key, sorted(declared)))
            chosen = "test" if "test" in declared else "valid"
        else:
            chosen = declared.pop()
        for row in members:
            row["split"] = chosen
        honoured += len(members)

    # Stratify only the groups that still need a split.
    strata = defaultdict(list)
    for key in unassigned:
        members = groups[key]
        strata[(members[0]["dataset"], members[0]["is_math"])].append(key)

    for stratum_keys in strata.values():
        for key in sorted(stratum_keys):
            bucket = stable_bucket(f"{seed}:{key}")
            if bucket < val_cut:
                split = "valid"
            elif bucket < test_cut:
                split = "test"
            else:
                split = "train"
            for row in groups[key]:
                row["split"] = split

    if honoured:
        print(f"[*] Honoured the upstream split column for {honoured} row(s)")
    if conflicts:
        print(f"[!] {len(conflicts)} label(s) were declared in conflicting upstream splits; "
              f"routed away from train:")
        for key, declared in conflicts[:5]:
            print(f"      {key[:60]!r} declared as {declared}")

    return rows, groups


def summarise(rows, groups):
    print()
    print("=" * 78)
    print(" SPLIT MANIFEST SUMMARY")
    print("=" * 78)

    by_split = Counter(r["split"] for r in rows)
    total = len(rows)
    print(f"\n{'split':<10}{'rows':>7}{'pct':>8}{'math':>8}{'text':>8}{'ctc-infeasible':>17}")
    print("-" * 78)
    for split in ("train", "valid", "test"):
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            continue
        math_n = sum(r["is_math"] for r in subset)
        bad = sum(1 for r in subset if not r["ctc_feasible"])
        print(f"{split:<10}{len(subset):>7}{100 * len(subset) / total:>7.1f}%"
              f"{math_n:>8}{len(subset) - math_n:>8}{bad:>10} ({100 * bad / len(subset):>4.1f}%)")
    print("-" * 78)
    print(f"{'TOTAL':<10}{total:>7}")

    print(f"\nDistinct label strings: {len(groups)} across {total} rows"
          f"  ({total - len(groups)} row(s) share a label with another row)")

    print(f"\n{'dataset':<18}{'train':>8}{'valid':>8}{'test':>8}{'infeasible':>13}")
    print("-" * 78)
    for dataset in DATASET_DIRS:
        subset = [r for r in rows if r["dataset"] == dataset]
        if not subset:
            continue
        counts = Counter(r["split"] for r in subset)
        bad = sum(1 for r in subset if not r["ctc_feasible"])
        print(f"{dataset:<18}{counts['train']:>8}{counts['valid']:>8}{counts['test']:>8}"
              f"{bad:>8} ({100 * bad / len(subset):>4.1f}%)")

    # The check that matters: no label may cross a split boundary.
    leaked = [key for key, members in groups.items()
              if len({m["split"] for m in members}) > 1]
    print()
    if leaked:
        print(f"[FAIL] {len(leaked)} label(s) appear in more than one split!")
        for key in leaked[:5]:
            print(f"        {key!r}")
    else:
        print("[OK] No label string appears in more than one split - no leakage.")

    infeasible = sum(1 for r in rows if not r["ctc_feasible"])
    if infeasible:
        print(f"[!] {infeasible} row(s) ({100 * infeasible / total:.1f}%) have a label longer than the")
        print("    available CTC timesteps. These cannot be learned and contribute no")
        print("    gradient. Train with --exclude-infeasible (the default) to drop them.")


def build_manifest(
    data_root: str = None,
    out_csv: str = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
    target_h: int = 32,
):
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    data_root = data_root or os.path.join(project_root, "data")
    out_csv = out_csv or os.path.join(data_root, "splits.csv")

    print(f"[*] Scanning datasets under: {data_root}")
    rows = collect_rows(data_root, target_h=target_h)
    if not rows:
        raise SystemExit(f"[!] No annotated rows found under {data_root}. Nothing to split.")

    rows, groups = assign_splits(rows, val_frac=val_frac, test_frac=test_frac, seed=seed)
    rows.sort(key=lambda r: (r["dataset"], r["image_path"]))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows, groups)
    print(f"\n[OK] Manifest written to: {out_csv}")
    print("     Every script now reads this file, so runs are comparable.")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the canonical OCR train/val/test manifest")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to the data/ directory (default: <project>/data)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output manifest path (default: <data-root>/splits.csv)")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337,
                        help="Changing this reshuffles the split; keep it fixed to stay comparable")
    parser.add_argument("--target-h", type=int, default=32,
                        help="Must match the model's input height, used to compute timesteps")
    args = parser.parse_args()

    build_manifest(
        data_root=args.data_root,
        out_csv=args.out,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        target_h=args.target_h,
    )
