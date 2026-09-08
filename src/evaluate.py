"""
Benchmark the CNN-BiLSTM recogniser against the TrOCR baseline.

What changed, and why each one mattered
---------------------------------------
1. Evaluates the held-out TEST split from data/splits.csv. The old default was
   `data/expanded` with `split=None`, i.e. the whole directory - all of which
   `train.py` had trained on. Every previously reported CER was measured on
   training data.

2. A missing CER no longer counts as 0.0. Lines 58 and 66 of the old version
   read `cnn_info["cer"] if ... is not None else 0.0`, so any sample that
   failed to produce a metric was recorded as a PERFECT score and averaged in.
   Failures now surface as failures.

3. The engine is constructed with strict=True. Previously, if the checkpoint
   was missing or mismatched, `infer.py` printed a warning and benchmarked
   randomly initialised weights.

4. Post-processing is off by default. Autocorrect against a ~95-word domain
   dictionary and LaTeX reformatting both change the string being scored, so
   the old numbers measured "model + dictionary + formatter", and a gain from
   any one of them looked like a model improvement. Use --postprocess to
   measure the full pipeline as a separate, clearly labelled row.

5. Results are broken down by dataset and by math/text, and every single
   prediction is written to a per-sample CSV. One aggregate number is what let
   "76% of the math data cannot be represented under CTC" stay invisible for
   two months. Read the per-sample file.

6. Reports median alongside mean, and counts empty predictions. A mean CER can
   sit at a plausible-looking 0.5 while half the outputs are empty strings.

7. TrOCR's parameter count is measured, not the hardcoded 62,000,000.

8. Removed the auto-call to `generate_expanded_dataset` on an empty split.
   Silently synthesising data in the middle of an evaluation run is how a
   benchmark ends up measuring something nobody chose.

Usage
-----
    python -m src.evaluate                          # test split, raw decode
    python -m src.evaluate --postprocess            # full pipeline
    python -m src.evaluate --model_type cnn --max_samples 50
"""

import argparse
import csv
import os
import statistics

import cv2

from src.dataset import DEFAULT_MANIFEST, OCRDataset
from src.infer import OCRInferenceEngine
from src.utils.metrics import OCRMetrics


def _stats(values):
    """Return (mean, median, n) ignoring None entries."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, 0
    return sum(clean) / len(clean), statistics.median(clean), len(clean)


def _print_block(title, params, cer_list, wer_list, lat_list, empty, skipped, n_total):
    mean_cer, med_cer, n_cer = _stats(cer_list)
    mean_wer, med_wer, _ = _stats(wer_list)
    mean_lat, _, _ = _stats(lat_list)

    print("\n" + "=" * 70)
    print(f" {title}")
    print("-" * 70)
    print(f" Parameters              : {params:,} (~{params / 1e6:.1f}M)")
    if mean_lat:
        print(f" Avg latency / image     : {mean_lat:.2f} ms  ({1000.0 / mean_lat:.1f} FPS)")
    if n_cer:
        print(f" CER  mean / median      : {mean_cer:.4f} / {med_cer:.4f}")
        print(f" WER  mean / median      : {mean_wer:.4f} / {med_wer:.4f}")
        print(f" Scored samples          : {n_cer} of {n_total}")
    else:
        print(" CER / WER               : no samples produced a metric")
    print(f" Empty predictions       : {empty} ({100.0 * empty / max(1, n_total):.1f}%)")
    if skipped:
        print(f" [!] Samples with no metric (NOT counted as 0.0): {skipped}")
    print("=" * 70)
    return mean_cer, mean_wer, mean_lat


def _breakdown(rows, key_fn, label, model_key):
    """Group per-sample rows and print mean CER per group."""
    groups = {}
    for row in rows:
        cer = row.get(f"{model_key}_cer")
        if cer is None:
            continue
        groups.setdefault(key_fn(row), []).append(cer)
    if len(groups) <= 1:
        return
    print(f"\n  CER by {label}:")
    for name in sorted(groups, key=lambda k: str(k)):
        vals = groups[name]
        print(f"    {str(name):<18} n={len(vals):<5} mean CER {sum(vals) / len(vals):.4f}")


def run_evaluation(
    manifest: str = None,
    data_dir: str = None,
    output_csv: str = None,
    per_sample_csv: str = None,
    model_type: str = "comparison",
    split: str = "test",
    max_samples: int = None,
    include_math: bool = True,
    exclude_infeasible: bool = False,
    postprocess: bool = False,
    use_tta: bool = False,
    trocr_model: str = None,
    strict: bool = True,
    checkpoint: str = None,
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_csv = output_csv or os.path.join(project_root, "docs", "benchmark_results.csv")
    per_sample_csv = per_sample_csv or os.path.join(project_root, "docs", "benchmark_per_sample.csv")
    model_type = (model_type or "comparison").lower()
    eval_mode = "both" if model_type in ("comparison", "both") else model_type

    print("=" * 70)
    print(f" MODEL EVALUATION - mode: {model_type.upper()}, split: {str(split).upper()}")
    print("=" * 70)
    print(f"  post-processing : {'ON (measuring model + dictionary + formatter)' if postprocess else 'OFF (measuring the model)'}")
    print(f"  TTA             : {use_tta}")
    print(f"  math samples    : {'included' if include_math else 'excluded'}")

    if data_dir:
        dataset = OCRDataset(data_dir=data_dir, split=split)
        source = data_dir
    else:
        dataset = OCRDataset(
            manifest=manifest or DEFAULT_MANIFEST, split=split,
            include_math=include_math, exclude_infeasible=exclude_infeasible,
        )
        source = manifest or DEFAULT_MANIFEST

    if len(dataset) == 0:
        raise SystemExit(
            f"[!] No samples in split {split!r} of {source}.\n"
            f"    Build the manifest first:  python -m src.make_splits"
        )

    n_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    print(f"  samples         : {n_total} from {source}")

    engine_kwargs = dict(strict=strict, load_trocr=eval_mode in ("transformer", "both"))
    if checkpoint:
        engine_kwargs["checkpoint_path"] = checkpoint
        print(f"  checkpoint      : {checkpoint}")
    if trocr_model:
        engine_kwargs["trocr_model"] = trocr_model
        print(f"  TrOCR checkpoint: {trocr_model}")
    engine = OCRInferenceEngine(**engine_kwargs)

    per_sample = []
    cnn_cer, cnn_wer, cnn_lat = [], [], []
    tr_cer, tr_wer, tr_lat = [], [], []
    cnn_empty = tr_empty = cnn_skip = tr_skip = 0

    for idx in range(n_total):
        img_path, ground_truth = dataset.samples[idx]
        meta = dataset.meta[idx] if idx < len(dataset.meta) else {}
        img_np = cv2.imread(img_path)
        if img_np is None:
            print(f"[!] Unreadable image, skipping: {img_path}")
            continue

        res = engine.run_pipeline(
            img_np, model_type=eval_mode, ground_truth=ground_truth,
            use_tta=use_tta, postprocess=postprocess,
        )

        row = {
            "image_path": os.path.relpath(img_path, project_root),
            "dataset": meta.get("dataset", ""),
            "is_math": int(bool(meta.get("is_math", False))),
            "ctc_feasible": int(bool(meta.get("ctc_feasible", True))),
            "ground_truth": ground_truth,
            "cnn_prediction": "", "cnn_cer": None, "cnn_wer": None,
            "trocr_prediction": "", "trocr_cer": None, "trocr_wer": None,
        }

        info = res.get("cnn_bilstm_attention")
        if info:
            row["cnn_prediction"] = info["predicted_text"]
            row["cnn_cer"] = info["cer"]
            row["cnn_wer"] = info.get("wer")
            if info["cer"] is None:
                cnn_skip += 1
            else:
                cnn_cer.append(info["cer"])
                cnn_wer.append(row["cnn_wer"])
            cnn_lat.append(info["latency_ms"])
            if not info["predicted_text"].strip():
                cnn_empty += 1

        info = res.get("transformer_baseline")
        if info:
            row["trocr_prediction"] = info["predicted_text"]
            row["trocr_cer"] = info["cer"]
            row["trocr_wer"] = info.get("wer")
            if info["cer"] is None:
                tr_skip += 1
            else:
                tr_cer.append(info["cer"])
                tr_wer.append(row["trocr_wer"])
            tr_lat.append(info["latency_ms"])
            if not info["predicted_text"].strip():
                tr_empty += 1

        per_sample.append(row)

        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            print(f"  [{idx + 1}/{n_total}]", flush=True)

    # ---------------------------------------------------------------- output
    os.makedirs(os.path.dirname(per_sample_csv), exist_ok=True)
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample)

    summary_rows = []
    cnn_params = OCRMetrics.count_parameters(engine.cnn_bilstm_model)

    if eval_mode in ("cnn", "both") and cnn_lat:
        if not engine.is_trained:
            print("\n[!] WARNING: the CNN ran on randomly initialised weights. "
                  "The numbers below are noise.")
        mean_cer, mean_wer, mean_lat = _print_block(
            "CNN + BiLSTM + Attention", cnn_params, cnn_cer, cnn_wer, cnn_lat,
            cnn_empty, cnn_skip, n_total,
        )
        _breakdown(per_sample, lambda r: r["dataset"], "dataset", "cnn")
        _breakdown(per_sample, lambda r: "math" if r["is_math"] else "text", "content type", "cnn")
        _breakdown(per_sample, lambda r: "feasible" if r["ctc_feasible"] else "CTC-infeasible",
                   "CTC feasibility", "cnn")
        summary_rows.append(["CNN + BiLSTM + Attention", cnn_params, round(mean_lat, 2),
                             round(1000.0 / mean_lat, 1) if mean_lat else 0.0,
                             round(mean_cer, 4) if mean_cer is not None else "",
                             round(mean_wer, 4) if mean_wer is not None else "",
                             len(cnn_cer), cnn_empty])

    if eval_mode in ("transformer", "both") and tr_lat:
        tr_params = engine.trocr.n_params if engine.trocr else 0
        mean_cer, mean_wer, mean_lat = _print_block(
            f"Vision Transformer ({engine.trocr.model_name if engine.trocr else 'TrOCR'})",
            tr_params, tr_cer, tr_wer, tr_lat, tr_empty, tr_skip, n_total,
        )
        _breakdown(per_sample, lambda r: r["dataset"], "dataset", "trocr")
        _breakdown(per_sample, lambda r: "math" if r["is_math"] else "text", "content type", "trocr")
        summary_rows.append([f"Vision Transformer ({engine.trocr.model_name if engine.trocr else 'TrOCR'})",
                             tr_params, round(mean_lat, 2),
                             round(1000.0 / mean_lat, 1) if mean_lat else 0.0,
                             round(mean_cer, 4) if mean_cer is not None else "",
                             round(mean_wer, 4) if mean_wer is not None else "",
                             len(tr_cer), tr_empty])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Architecture", "Parameters Count", "Avg Latency (ms)",
                         "Throughput (FPS)", "CER", "WER", "Scored Samples", "Empty Predictions"])
        writer.writerows(summary_rows)

    print(f"\n[OK] Summary     : {output_csv}")
    print(f"[OK] Per-sample  : {per_sample_csv}")
    print("\nOpen the per-sample file and read the ground_truth / prediction columns "
          "side by side.\nThe aggregate CER tells you how much is wrong; only that file "
          "tells you what is wrong.")

    worst = sorted((r for r in per_sample if r["cnn_cer"] is not None),
                   key=lambda r: -r["cnn_cer"])[:5]
    if worst:
        print("\nWorst 5 CNN predictions:")
        for r in worst:
            print(f"  CER {r['cnn_cer']:.2f}  {r['ground_truth'][:38]!r} -> {r['cnn_prediction'][:38]!r}")

    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the OCR models on a held-out split")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", type=str, default=None,
                        help="Legacy directory mode; bypasses the manifest and its leakage guarantees")
    parser.add_argument("--output_csv", "--output-csv", dest="output_csv", type=str, default=None)
    parser.add_argument("--per_sample_csv", "--per-sample-csv", dest="per_sample_csv",
                        type=str, default=None)
    parser.add_argument("--model_type", "--model-type", dest="model_type", type=str,
                        default="comparison", choices=["cnn", "transformer", "comparison", "both"])
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to score (default: test - the only honest choice)")
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--no-math", dest="include_math", action="store_false",
                        help="Exclude LaTeX math samples")
    parser.add_argument("--exclude-infeasible", action="store_true",
                        help="Exclude samples the CNN cannot represent under CTC")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply autocorrect / LaTeX formatting before scoring")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--trocr-model", type=str, default=None,
                        help="e.g. microsoft/trocr-base-handwritten for handwritten data")
    parser.add_argument("--allow-untrained", dest="strict", action="store_false",
                        help="Permit running without a valid checkpoint (produces meaningless numbers)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Score a specific checkpoint instead of "
                             "src/models/checkpoints/cnn_bilstm_best.pth. Needed to "
                             "compare two training runs, e.g. the SSL-init ablation.")
    args = parser.parse_args()

    run_evaluation(
        manifest=args.manifest,
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        per_sample_csv=args.per_sample_csv,
        model_type=args.model_type,
        split=args.split,
        max_samples=args.max_samples,
        include_math=args.include_math,
        exclude_infeasible=args.exclude_infeasible,
        postprocess=args.postprocess,
        use_tta=args.use_tta,
        trocr_model=args.trocr_model,
        strict=args.strict,
        checkpoint=args.checkpoint,
    )
