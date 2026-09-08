"""
High-Accuracy Text Recognition & Extraction Tool
Supports single line crops and full document pages using:
1. Microsoft Vision Transformer (TrOCR) - High accuracy Transformer SOTA
2. EasyOCR - Robust multi-line document & scene text engine
3. Primary Custom CNN + BiLSTM + Attention - Project research architecture
"""

import os
import sys
import argparse
import cv2
import numpy as np

def run_extraction(image_path: str, model_choice: str = "all", is_document: bool = False):
    if not os.path.exists(image_path):
        print(f"[X] Image file not found: {image_path}")
        return

    print("=" * 65)
    print(" HIGH-ACCURACY TEXT RECOGNITION SYSTEM")
    print("=" * 65)
    print(f" Input Image : {image_path}")
    print(f" Model Mode  : {model_choice.upper()}")
    print("-" * 65)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[X] Could not read image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f" Resolution  : {w}x{h} px")

    # 1. Full Page Document OCR
    if is_document or h > 150:
        print("\n[*] Processing as Multi-Line Document...")
        import easyocr
        reader = easyocr.Reader(["en"], verbose=False)
        results = reader.readtext(img)
        print("\n" + "=" * 65)
        print(" EXTRACTED DOCUMENT TEXT (Multi-Line):")
        print("=" * 65)
        from src.utils.postprocessing import OCRPostProcessor
        for idx, (bbox, text, conf) in enumerate(results, 1):
            corr_text = OCRPostProcessor.process(text, enable_autocorrect=True)
            print(f" Line {idx:02d} [{conf*100:5.1f}%]: {corr_text}")
            full_text.append(corr_text)
        print("-" * 65)
        combined_text = OCRPostProcessor.process("\n".join(full_text), enable_autocorrect=True)
        print("\nCombined Transcript:\n" + combined_text)
        print("=" * 65)
        return

    # 2. Line Crop Text Extraction
    from src.utils.postprocessing import OCRPostProcessor
    if model_choice in ("trocr", "all"):
        print("\n[1] Running Microsoft Vision Transformer (TrOCR)...")
        from src.infer import OCRInferenceEngine
        engine = OCRInferenceEngine(load_trocr=True)
        raw_res, _ = engine.predict_transformer(img, enable_autocorrect=False)
        corr_res, _ = engine.predict_transformer(img, enable_autocorrect=True)
        raw_txt = raw_res.get('predicted_text', '').strip()
        corr_txt = corr_res.get('predicted_text', '').strip()
        print(f"  --> TrOCR Raw    : {raw_txt!r}")
        print(f"  --> Auto-Correct : {corr_txt!r}")
        print(f"  --> Latency      : {corr_res.get('latency_ms', 0):.2f} ms")

    if model_choice in ("easyocr", "all"):
        print("\n[2] Running EasyOCR Engine...")
        import easyocr
        reader = easyocr.Reader(["en"], verbose=False)
        texts = reader.readtext(img, detail=0)
        raw_easy = " ".join(texts)
        corr_easy = OCRPostProcessor.process(raw_easy, enable_autocorrect=True)
        print(f"  --> EasyOCR Raw  : {raw_easy!r}")
        print(f"  --> Auto-Correct : {corr_easy!r}")

    if model_choice in ("cnn", "all"):
        print("\n[3] Running Custom CNN + BiLSTM + Attention...")
        from src.infer import OCRInferenceEngine
        engine = OCRInferenceEngine(load_trocr=False)
        raw_res, _ = engine.predict_cnn(img, enable_autocorrect=False)
        corr_res, _ = engine.predict_cnn(img, enable_autocorrect=True)
        raw_txt = raw_res.get('predicted_text', '').strip()
        corr_txt = corr_res.get('predicted_text', '').strip()
        print(f"  --> CNN Raw      : {raw_txt!r}")
        print(f"  --> Auto-Correct : {corr_txt!r}")

    print("\n" + "=" * 65)
    print(" [OK] Extraction Finished.")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text with high accuracy")
    parser.add_argument("--image", type=str, default="data/samples/printed_sample.png",
                        help="Path to image file (default: data/samples/printed_sample.png)")
    parser.add_argument("--model", type=str, default="all", choices=["trocr", "easyocr", "cnn", "all"],
                        help="Model to use for extraction (default: all)")
    parser.add_argument("--doc", action="store_true", help="Process as full multi-line document")
    args = parser.parse_args()

    run_extraction(args.image, model_choice=args.model, is_document=args.doc)
