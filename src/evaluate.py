import os
import argparse
import csv
import time
import cv2
import torch
import numpy as np

from src.dataset import OCRDataset
from src.infer import OCRInferenceEngine
from src.utils.metrics import OCRMetrics

def run_evaluation(
    data_dir: str = "d:\\Major Project\\data\\expanded",
    output_csv: str = "d:\\Major Project\\docs\\benchmark_results.csv",
    model_type: str = "comparison"
):
    model_type = model_type.lower()
    print("=" * 70)
    print(f" [*] STARTING MODEL EVALUATION (Mode: {model_type.upper()})")
    print("=" * 70)

    # 1. Load Dataset
    dataset = OCRDataset(data_dir=data_dir)
    if len(dataset) == 0:
        print(f"[!] No evaluation samples found in '{data_dir}'. Generating dataset first...")
        from data.dataset_expander import generate_expanded_dataset
        generate_expanded_dataset(output_dir=data_dir, num_samples=100)
        dataset = OCRDataset(data_dir=data_dir)

    print(f"[*] Loaded {len(dataset)} evaluation samples from '{data_dir}'")

    # 2. Instantiate Inference Engine
    engine = OCRInferenceEngine()

    cnn_cer_list, cnn_wer_list, cnn_latency_list = [], [], []
    trocr_cer_list, trocr_wer_list, trocr_latency_list = [], [], []

    eval_mode = "both" if model_type in ["comparison", "both"] else model_type

    # Process evaluation loop
    for idx in range(len(dataset)):
        img_path, ground_truth = dataset.samples[idx]
        img_np = cv2.imread(img_path)
        if img_np is None:
            continue

        res = engine.run_pipeline(img_np, model_type=eval_mode, ground_truth=ground_truth)

        # Extract Primary CNN Metrics if active
        if res.get("cnn_bilstm_attention"):
            cnn_info = res["cnn_bilstm_attention"]
            cnn_cer_list.append(cnn_info["cer"] if cnn_info["cer"] is not None else 0.0)
            cnn_wer = OCRMetrics.calculate_wer(ground_truth, cnn_info["predicted_text"])
            cnn_wer_list.append(cnn_wer)
            cnn_latency_list.append(cnn_info["latency_ms"])

        # Extract TrOCR Baseline Metrics if active
        if res.get("transformer_baseline"):
            trocr_info = res["transformer_baseline"]
            trocr_cer_list.append(trocr_info["cer"] if trocr_info["cer"] is not None else 0.0)
            trocr_wer = OCRMetrics.calculate_wer(ground_truth, trocr_info["predicted_text"])
            trocr_wer_list.append(trocr_wer)
            trocr_latency_list.append(trocr_info["latency_ms"])

    cnn_params = OCRMetrics.count_parameters(engine.cnn_bilstm_model)
    trocr_params = 62000000

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Architecture", "Parameters Count", "Avg Latency (ms)", "Throughput (FPS)", "CER", "WER"])

        if model_type in ["cnn", "comparison", "both"] and cnn_latency_list:
            cnn_avg_cer = round(sum(cnn_cer_list) / max(1, len(cnn_cer_list)), 4)
            cnn_avg_wer = round(sum(cnn_wer_list) / max(1, len(cnn_wer_list)), 4)
            cnn_avg_lat = round(sum(cnn_latency_list) / max(1, len(cnn_latency_list)), 2)
            cnn_fps = round(1000.0 / cnn_avg_lat, 1) if cnn_avg_lat > 0 else 0.0
            
            writer.writerow(["CNN + BiLSTM + Attention", cnn_params, cnn_avg_lat, cnn_fps, cnn_avg_cer, cnn_avg_wer])
            
            print("\n" + "=" * 65)
            print(" CNN + BiLSTM + Attention Model Evaluation Metrics:")
            print("-" * 65)
            print(f" Parameters Count:          {cnn_params:,} (~{round(cnn_params/1e6, 2)}M)")
            print(f" Avg Latency per Image:     {cnn_avg_lat} ms")
            print(f" Inference Throughput:      {cnn_fps} FPS")
            print(f" Character Error Rate (CER):{cnn_avg_cer:.4f}")
            print(f" Word Error Rate (WER):     {cnn_avg_wer:.4f}")
            print("=" * 65)

        if model_type in ["transformer", "comparison", "both"] and trocr_latency_list:
            trocr_avg_cer = round(sum(trocr_cer_list) / max(1, len(trocr_cer_list)), 4)
            trocr_avg_wer = round(sum(trocr_wer_list) / max(1, len(trocr_wer_list)), 4)
            trocr_avg_lat = round(sum(trocr_latency_list) / max(1, len(trocr_latency_list)), 2)
            trocr_fps = round(1000.0 / trocr_avg_lat, 1) if trocr_avg_lat > 0 else 0.0

            writer.writerow(["Vision Transformer (TrOCR)", trocr_params, trocr_avg_lat, trocr_fps, trocr_avg_cer, trocr_avg_wer])

            print("\n" + "=" * 65)
            print(" Vision Transformer (TrOCR) Model Evaluation Metrics:")
            print("-" * 65)
            print(f" Parameters Count:          {trocr_params:,} (~62M)")
            print(f" Avg Latency per Image:     {trocr_avg_lat} ms")
            print(f" Inference Throughput:      {trocr_fps} FPS")
            print(f" Character Error Rate (CER):{trocr_avg_cer:.4f}")
            print(f" Word Error Rate (WER):     {trocr_avg_wer:.4f}")
            print("=" * 65)

    print(f"\n[OK] Evaluation completed! Results saved to '{output_csv}'")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR Models Benchmark")
    parser.add_argument("--data_dir", type=str, default="d:\\Major Project\\data\\expanded")
    parser.add_argument("--output_csv", type=str, default="d:\\Major Project\\docs\\benchmark_results.csv")
    parser.add_argument("--model_type", type=str, default="comparison", choices=["cnn", "transformer", "comparison", "both"])
    args = parser.parse_args()

    run_evaluation(data_dir=args.data_dir, output_csv=args.output_csv, model_type=args.model_type)
