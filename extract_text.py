"""
High-Accuracy Text Recognition & Extraction Tool
Supports all types of documents including:
  - Printed text documents
  - Handwritten cursive manuscripts
  - Mathematical LaTeX equation documents (derivatives, fractions, integrals, symbols)
  - Historical archival scans (degraded tone, Gaelic/Irish lexicon)
  - Full multi-line document pages

Models Supported:
  1. Primary Architecture: CNN + BiLSTM + Attention (FP32 & Edge INT8 Quantized)
  2. Baseline Architecture: Microsoft Vision Transformer (TrOCR)
  3. Mathematical Engine: 2D Visual Layout Decomposition + LaTeX Synthesizer
  4. Optional: EasyOCR
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.infer import OCRInferenceEngine
from src.document_ocr import DocumentOCREngine
from src.utils.math_recognizer import MathFormulaParser
from src.utils.postprocessing import OCRPostProcessor


def run_extraction(
    image_path: str,
    model_choice: str = "both",
    domain: str = "auto",
    is_document: bool = False,
    use_quantized: bool = False,
    is_historical: bool = False,
):
    if not os.path.exists(image_path):
        print(f"\n[X] Error: Image file not found at '{image_path}'")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[X] Error: Could not decode image at '{image_path}'")
        return

    h, w = img.shape[:2]
    model_choice = model_choice.lower()
    if model_choice == "all":
        model_choice = "both"

    print("\n" + "=" * 75)
    print("        HIGH-ACCURACY TEXT & MATHEMATICAL EQUATION EXTRACTION")
    print("=" * 75)
    print(f" File Path     : {image_path}")
    print(f" Resolution    : {w} x {h} px")
    print(f" Target Model  : {model_choice.upper()} (Quantized INT8: {use_quantized})")

    # Detect visual structure and domain
    has_math_structure = MathFormulaParser.has_math_visual_structure(img)
    is_math_domain = domain == "math" or has_math_structure
    effective_domain = "math" if is_math_domain else (domain if domain != "auto" else "printed")

    is_multi_line = is_document or (h > 140 and w > 180 and not is_math_domain)
    doc_type_label = (
        "Multi-Line Document Scan"
        if is_multi_line
        else ("Mathematical Equation Document" if is_math_domain else ("Historical Document" if is_historical else "Single Line / Crop"))
    )
    print(f" Document Type : {doc_type_label}")
    print("-" * 75)

    # -------------------------------------------------------------------------
    # 1. Full-Page Multi-Line Document Mode
    # -------------------------------------------------------------------------
    if is_multi_line:
        load_trocr = model_choice in ("transformer", "both", "all")
        inf_engine = OCRInferenceEngine(load_trocr=load_trocr, use_quantized=use_quantized)
        doc_engine = DocumentOCREngine(inference_engine=inf_engine, use_quantized=use_quantized)
        start_t = time.perf_counter()
        doc_res = doc_engine.process_document(
            img,
            model_type=model_choice,
            enable_autocorrect=True,
            domain=domain,
            is_historical=is_historical,
            use_quantized=use_quantized,
        )
        total_time = (time.perf_counter() - start_t) * 1000.0

        total_lines = doc_res["total_lines_detected"]
        print(f"[OK] Successfully segmented {total_lines} line(s) in {total_time:.1f} ms.\n")

        print("=" * 75)
        print(" LINE-BY-LINE RECOGNITION BREAKDOWN:")
        print("=" * 75)
        for item in doc_res["lines"]:
            ln = item["line_num"]
            cat = "MATH" if item.get("is_math") else "TEXT"
            if model_choice in ("both", "all"):
                print(f" Line {ln:02d} [{cat:4s}] | [CNN]: {item['cnn_text'] or '<none>'}")
                print(f"               | [TrOCR]: {item['trocr_text'] or '<none>'}")
            elif model_choice == "cnn":
                print(f" Line {ln:02d} [{cat:4s}] | CNN: {item['cnn_text'] or '<none>'}")
            elif model_choice == "transformer":
                print(f" Line {ln:02d} [{cat:4s}] | TrOCR: {item['trocr_text'] or '<none>'}")
        print("-" * 75)

        print("\n" + "=" * 75)
        print(" EXTRACTED DOCUMENT TRANSCRIPT OUTPUT:")
        print("=" * 75)
        if model_choice in ("cnn", "both"):
            print("\n--- [1] Primary Model (CNN + BiLSTM + Attention) Transcript ---")
            print(doc_res["cnn_transcript"] or "<No text extracted>")
        if model_choice in ("transformer", "both"):
            print("\n--- [2] Vision Transformer (TrOCR) Transcript ---")
            print(doc_res["trocr_transcript"] or "<No text extracted>")
        if doc_res.get("best_transcript"):
            print("\n--- [*] Best Combined / Unified Transcript ---")
            print(doc_res["best_transcript"])
        print("=" * 75)
        return

    # -------------------------------------------------------------------------
    # 2. Single Crop / Line / Equation Document Mode
    # -------------------------------------------------------------------------
    load_trocr = model_choice in ("transformer", "both")
    engine = OCRInferenceEngine(load_trocr=load_trocr, use_quantized=use_quantized)

    pipeline_res = engine.run_pipeline(
        img,
        model_type=model_choice,
        enable_autocorrect=True,
        domain=effective_domain,
        is_historical=is_historical,
    )

    cnn_info = pipeline_res.get("cnn_bilstm_attention")
    trocr_info = pipeline_res.get("transformer_baseline")
    math_info = pipeline_res.get("math_engine")

    print("\n" + "=" * 75)
    print(" EXTRACTED TEXT OUTPUT FROM DOCUMENT:")
    print("=" * 75)

    if is_math_domain and math_info:
        print("\n [* MATHEMATICAL EQUATION ENGINE]")
        latex_text = math_info.get("predicted_text", "")
        print(f"  --> Canonical LaTeX Syntax : {latex_text}")
        print(f"  --> Stacked Fraction      : {'Yes (2D Layout Decomposition)' if math_info.get('has_fraction') else 'No (Linear/Operator)'}")
        print(f"  --> Latency               : {math_info.get('latency_ms', 0):.2f} ms")

    if cnn_info:
        model_name = cnn_info.get("model_name", "CNN + BiLSTM + Attention")
        pred_text = cnn_info.get("predicted_text", "")
        print(f"\n [MODEL 1: {model_name}]")
        print(f"  --> Extracted Output      : {pred_text}")
        print(f"  --> Latency               : {cnn_info.get('latency_ms', 0):.2f} ms")
        print(f"  --> Parameter Count       : {cnn_info.get('parameters_count', 0) / 1e6:.2f} M")
        print(f"  --> Is Mathematical      : {cnn_info.get('is_math', False)}")

    if trocr_info:
        model_name = trocr_info.get("model_name", "Vision Transformer (TrOCR)")
        pred_text = trocr_info.get("predicted_text", "")
        print(f"\n [MODEL 2: {model_name}]")
        print(f"  --> Extracted Output      : {pred_text}")
        print(f"  --> Latency               : {trocr_info.get('latency_ms', 0):.2f} ms")
        print(f"  --> Parameter Count       : {trocr_info.get('parameters_count', 0) / 1e6:.1f} M")
        print(f"  --> Detected Domain       : {trocr_info.get('detected_domain', 'printed')}")

    print("\n" + "=" * 75)
    print(" [OK] Extraction Finished Successfully.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-Accuracy Text & Mathematical Recognition Tool"
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to image file (positional argument)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (default: data/samples/printed_sample.png)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["both", "cnn", "transformer", "trocr", "all"],
        help="OCR model architecture to run (default: both)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="auto",
        choices=["auto", "printed", "handwritten", "math", "historical"],
        help="Document domain type (default: auto)",
    )
    parser.add_argument(
        "--doc",
        action="store_true",
        help="Force multi-line document segmentation mode",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use edge INT8 quantized model for CNN inference",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Apply archival document background restoration & Irish lexicon",
    )

    args = parser.parse_args()
    target_image = args.image_pos or args.image or "data/samples/printed_sample.png"
    model_key = "transformer" if args.model == "trocr" else args.model

    run_extraction(
        target_image,
        model_choice=model_key,
        domain=args.domain,
        is_document=args.doc,
        use_quantized=args.quantized,
        is_historical=args.historical,
    )
