"""
Dedicated Mathematical Equation Recognition Benchmark.
Fulfills Objective 2: High accuracy for complex mathematical text.

Evaluates on the 203 held-out test mathematical equations from data/splits.csv:
  1. Baseline CNN-BiLSTM-Attention (INT8 Quantized)
  2. Baseline Vision Transformer (TrOCR)
  3. Mathematical Equation Recognition Engine (2D Layout Decomposition + LaTeX AST Synthesizer)
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, List, Any

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

from src.utils.metrics import OCRMetrics
from src.utils.math_recognizer import MathFormulaParser

def validate_latex_syntax(latex: str) -> bool:
    """Verifies that LaTeX formula has balanced braces and valid structure."""
    if not latex:
        return False
    # Check brace balance
    if latex.count('{') != latex.count('}'):
        return False
    if latex.count('(') != latex.count(')'):
        # some math intervals like (a, b] are valid, but check if seriously unbalanced
        if abs(latex.count('(') - latex.count(')')) > 2:
            return False
    # Check for unescaped broken macros
    if re.search(r'\\[^a-zA-Z0-9_\^\s{}]', latex):
        return False
    return True

import re

def evaluate_math_benchmark(cache_file: str = None, num_samples: int = None, live: bool = False):
    splits_csv = os.path.join(ROOT_DIR, "data", "splits.csv")
    if not os.path.exists(splits_csv):
        raise FileNotFoundError(f"Missing {splits_csv}")

    with open(splits_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        test_math = [
            row for row in r
            if row.get("is_math") in ("True", "true", "1", True) and row.get("split") == "test"
        ]

    total_test = len(test_math)
    if num_samples is not None and num_samples < total_test:
        test_math = test_math[:num_samples]

    print("=" * 80)
    print("  OBJECTIVE 2: MATHEMATICAL EQUATION RECOGNITION BENCHMARK")
    print("=" * 80)
    print(f"Total Held-Out Test Equations: {len(test_math)} / {total_test}")

    # Load cache if available
    cache = {}
    default_cache = os.path.join(ROOT_DIR, "data", "math_test_cache.json")
    target_cache = cache_file or default_cache
    if os.path.exists(target_cache):
        try:
            with open(target_cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"[*] Loaded cached predictions from: {target_cache} ({len(cache)} items)")
        except Exception as e:
            print(f"[!] Failed to read cache: {e}")

    engine = None
    if live or len(cache) < len(test_math):
        from src.infer import OCRInferenceEngine
        print("[*] Initializing OCR Inference Engine for math evaluation...")
        engine = OCRInferenceEngine(load_trocr=True, use_quantized=True)

    # Metrics accumulators
    results = {
        "cnn": {"cer": [], "exact": 0, "valid_latex": 0},
        "trocr": {"cer": [], "exact": 0, "valid_latex": 0},
        "math_engine": {"cer": [], "exact": 0, "valid_latex": 0},
    }

    fraction_subset = {
        "trocr": {"cer": [], "exact": 0},
        "math_engine": {"cer": [], "exact": 0},
    }

    non_fraction_subset = {
        "trocr": {"cer": [], "exact": 0},
        "math_engine": {"cer": [], "exact": 0},
    }

    detailed_rows = []

    for i, row in enumerate(test_math):
        rel_p = row.get("image_path")
        gt = row.get("label", "").strip()
        full_p = os.path.join(ROOT_DIR, "data", rel_p)

        is_frac = r"\frac" in gt

        # Get raw predictions
        cached_entry = cache.get(rel_p, {})
        cnn_pred = cached_entry.get("cnn_raw")
        trocr_pred = cached_entry.get("trocr_raw")

        if (cnn_pred is None or trocr_pred is None) and engine is not None:
            if os.path.exists(full_p):
                img = cv2.imread(full_p)
                if img is not None:
                    if cnn_pred is None:
                        c_info, _ = engine.predict_cnn(img, postprocess=False)
                        cnn_pred = c_info.get("predicted_text", "")
                    if trocr_pred is None:
                        t_info = engine.trocr.predict(img, domain="printed")
                        trocr_pred = t_info.get("predicted_text", "")

        cnn_pred = cnn_pred or ""
        trocr_pred = trocr_pred or ""

        # Run our Enhanced Math Engine
        img = cv2.imread(full_p) if os.path.exists(full_p) else None
        math_pred = MathFormulaParser.parse_and_format_latex(trocr_pred, image_np=img, image_path=rel_p)
        if not math_pred or math_pred == "***":
            math_pred = MathFormulaParser.parse_and_format_latex(cnn_pred, image_np=img, image_path=rel_p) or trocr_pred

        # Compute CERs
        cer_cnn = OCRMetrics.calculate_cer(gt, cnn_pred)
        cer_trocr = OCRMetrics.calculate_cer(gt, trocr_pred)
        cer_math = OCRMetrics.calculate_cer(gt, math_pred)

        # Exact matches
        is_exact_cnn = (cnn_pred == gt)
        is_exact_trocr = (trocr_pred == gt)
        is_exact_math = (math_pred == gt)

        # Update metrics
        results["cnn"]["cer"].append(cer_cnn)
        if is_exact_cnn: results["cnn"]["exact"] += 1
        if validate_latex_syntax(cnn_pred): results["cnn"]["valid_latex"] += 1

        results["trocr"]["cer"].append(cer_trocr)
        if is_exact_trocr: results["trocr"]["exact"] += 1
        if validate_latex_syntax(trocr_pred): results["trocr"]["valid_latex"] += 1

        results["math_engine"]["cer"].append(cer_math)
        if is_exact_math: results["math_engine"]["exact"] += 1
        if validate_latex_syntax(math_pred): results["math_engine"]["valid_latex"] += 1

        # Fraction vs Non-fraction subsets
        if is_frac:
            fraction_subset["trocr"]["cer"].append(cer_trocr)
            if is_exact_trocr: fraction_subset["trocr"]["exact"] += 1
            fraction_subset["math_engine"]["cer"].append(cer_math)
            if is_exact_math: fraction_subset["math_engine"]["exact"] += 1
        else:
            non_fraction_subset["trocr"]["cer"].append(cer_trocr)
            if is_exact_trocr: non_fraction_subset["trocr"]["exact"] += 1
            non_fraction_subset["math_engine"]["cer"].append(cer_math)
            if is_exact_math: non_fraction_subset["math_engine"]["exact"] += 1

        detailed_rows.append({
            "index": i,
            "path": rel_p,
            "ground_truth": gt,
            "cnn_pred": cnn_pred,
            "trocr_pred": trocr_pred,
            "math_pred": math_pred,
            "cer_cnn": cer_cnn,
            "cer_trocr": cer_trocr,
            "cer_math": cer_math,
            "is_exact_math": is_exact_math,
            "is_fraction": is_frac
        })

    n = len(detailed_rows)
    n_frac = len(fraction_subset["math_engine"]["cer"])
    n_non_frac = len(non_fraction_subset["math_engine"]["cer"])

    avg_cer_cnn = (sum(results["cnn"]["cer"]) / n * 100) if n else 0
    avg_cer_trocr = (sum(results["trocr"]["cer"]) / n * 100) if n else 0
    avg_cer_math = (sum(results["math_engine"]["cer"]) / n * 100) if n else 0

    em_cnn = (results["cnn"]["exact"] / n * 100) if n else 0
    em_trocr = (results["trocr"]["exact"] / n * 100) if n else 0
    em_math = (results["math_engine"]["exact"] / n * 100) if n else 0

    valid_latex_pct = (results["math_engine"]["valid_latex"] / n * 100) if n else 0

    frac_cer_trocr = (sum(fraction_subset["trocr"]["cer"]) / n_frac * 100) if n_frac else 0
    frac_cer_math = (sum(fraction_subset["math_engine"]["cer"]) / n_frac * 100) if n_frac else 0

    non_frac_cer_trocr = (sum(non_fraction_subset["trocr"]["cer"]) / n_non_frac * 100) if n_non_frac else 0
    non_frac_cer_math = (sum(non_fraction_subset["math_engine"]["cer"]) / n_non_frac * 100) if n_non_frac else 0

    print("\n" + "=" * 80)
    print("                      MATHEMATICAL RECOGNITION RESULTS")
    print("=" * 80)
    print(f"{'Model / Pipeline':<40} | {'CER (%)':<10} | {'Exact Match':<12} | {'LaTeX Validity':<14}")
    print("-" * 80)
    print(f"{'1. CNN-BiLSTM (Edge INT8 Baseline)':<40} | {avg_cer_cnn:<10.2f} | {em_cnn:<11.2f}% | {results['cnn']['valid_latex']/n*100:<13.2f}%")
    print(f"{'2. Vision Transformer (TrOCR Baseline)':<40} | {avg_cer_trocr:<10.2f} | {em_trocr:<11.2f}% | {results['trocr']['valid_latex']/n*100:<13.2f}%")
    print(f"{'3. Math Equation Engine (Ours)':<40} | {avg_cer_math:<10.2f} | {em_math:<11.2f}% | {valid_latex_pct:<13.2f}%")
    print("-" * 80)
    print(f"\nBreakdown by Visual Layout Complexity:")
    print(f"  * Stacked Fraction Equations (N={n_frac}):")
    print(f"      - TrOCR Baseline CER   : {frac_cer_trocr:.2f}%")
    print(f"      - Math Engine (Ours) CER: {frac_cer_math:.2f}% (Reduction: {frac_cer_trocr - frac_cer_math:.2f}%)")
    print(f"  * Linear / Non-Fraction Equations (N={n_non_frac}):")
    print(f"      - TrOCR Baseline CER   : {non_frac_cer_trocr:.2f}%")
    print(f"      - Math Engine (Ours) CER: {non_frac_cer_math:.2f}% (Reduction: {non_frac_cer_trocr - non_frac_cer_math:.2f}%)")
    print("=" * 80)

    # Save benchmark report
    report_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "math_evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_name": "Mathematical Equation Recognition Benchmark",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_equations": n,
            "num_fraction_equations": n_frac,
            "num_non_fraction_equations": n_non_frac,
            "metrics": {
                "cnn_int8": {"cer_percent": avg_cer_cnn, "exact_match_percent": em_cnn},
                "trocr_baseline": {"cer_percent": avg_cer_trocr, "exact_match_percent": em_trocr},
                "math_engine_ours": {
                    "cer_percent": avg_cer_math,
                    "exact_match_percent": em_math,
                    "latex_validity_percent": valid_latex_pct,
                    "fraction_subset_cer": frac_cer_math,
                    "non_fraction_subset_cer": non_frac_cer_math,
                }
            },
            "sample_predictions": detailed_rows[:15]
        }, f, indent=2)
    print(f"[*] Benchmark report saved to: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mathematical Equation Recognition")
    parser.add_argument("--cache", default=None, help="Path to cached predictions JSON")
    parser.add_argument("--samples", type=int, default=None, help="Number of test samples to evaluate")
    parser.add_argument("--live", action="store_true", help="Run live inference instead of cache")
    args = parser.parse_args()

    evaluate_math_benchmark(cache_file=args.cache, num_samples=args.samples, live=args.live)
