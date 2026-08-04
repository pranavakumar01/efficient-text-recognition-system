# Efficient Text Recognition System (DL + SSL)

> **7th Semester Phase-I Major Project** • Dept. of Information Science & Engineering, NMAM Institute of Technology, Nitte

An end-to-end Optical Character Recognition (OCR) framework combining **Deep Learning (CNN + BiLSTM + Attention)** and **Self-Supervised Learning (SimCLR SSL)** with comparative evaluation against **Vision Transformers (TrOCR)**.

---

## 🌟 Key Features

- 🧠 **Primary Architecture**: CNN Feature Extractor + Bidirectional LSTM + Bahdanau Attention Decoder (CTC Loss).
- 🤖 **Baseline Architecture**: Pre-trained Vision Transformer (`microsoft/trocr-small-printed`).
- 🧬 **Self-Supervised Learning**: SimCLR Contrastive Representation Pre-training (`src/train_ssl.py`) for transfer learning without label dependencies.
- 📄 **Full-Page Document Segmentation**: Horizontal projection profile line segmenter (`src/document_ocr.py`) for multi-line document extraction.
- 🔤 **Post-Processing Engine**: Character error correction and dictionary-based token refining (`src/utils/postprocessing.py`).
- 📊 **Automated Report Figures**: Built-in chart generator (`src/utils/generate_charts.py`) producing empirical loss/CER curves and architectural comparison plots.
- 💻 **Modern Web UI & FastAPI Server**: Glassmorphism web interface (`app/index.html`) with live model selector, preprocessing stage visualizer, and transcript exporter (**TXT** / **JSON**).

---

## 📁 Repository Structure

```text
Major Project/
├── app/
│   ├── index.html           # Web Demo Interface
│   └── server.py            # FastAPI Backend API Server
├── data/
│   ├── generate_samples.py  # Evaluation Preset Generator
│   ├── generate_synthetic.py# Synthetic Dataset Generator
│   └── dataset_expander.py  # Multi-Domain Dataset Expander
├── docs/
│   ├── benchmark_results.csv# Comparative Performance Metrics
│   └── figures/             # Training Curves & Benchmark Charts
├── src/
│   ├── models/
│   │   ├── cnn_bilstm_att.py# Primary CNN + BiLSTM + Attention Model
│   │   ├── ssl_backbone.py  # SimCLR SSL Representation Module
│   │   └── trocr_baseline.py# Vision Transformer (TrOCR) Model
│   ├── preprocessing/       # Enhancement & ROI Normalization
│   ├── utils/               # Metrics, Visualizer, Postprocessing, Chart Generator
│   ├── dataset.py           # Dataset & CTC Collate Functions
│   ├── document_ocr.py      # Line Segmentation & Document OCR
│   ├── evaluate.py          # Benchmark Evaluation Script
│   ├── infer.py             # Inference Engine
│   ├── train.py             # Supervised CTC Training Pipeline
│   ├── train_ssl.py         # Self-Supervised Pre-Training Pipeline
│   └── train_transformer.py # Transformer Model Fine-Tuning Pipeline
├── run_demo.bat             # One-Click Web Demo Launcher
├── run_training.bat         # One-Click Model Training Pipeline
└── setup_environment.bat    # Virtual Environment Setup Script
```

---

## 🚀 Quick Start Guide

### 1. Launch Web Application
Double-click `run_demo.bat` or run:
```bash
.\venv\Scripts\activate
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### 2. Train Models Independently
```bash
# Supervised CTC Training with SSL Transfer Learning
python -m src.train --epochs 15 --use_ssl

# Vision Transformer (TrOCR) Fine-Tuning
python -m src.train_transformer --epochs 3

# SimCLR Contrastive Self-Supervised Pre-Training
python -m src.train_ssl --epochs 5
```

### 3. Run Benchmark Evaluation
```bash
# Evaluate CNN Model
python -m src.evaluate --model_type cnn

# Evaluate Vision Transformer Model
python -m src.evaluate --model_type transformer

# Comparative Benchmark (Both Models Side-by-Side)
python -m src.evaluate --model_type comparison
```

---

## 📄 License & Attribution
Major Project Phase-I (2025–2026) • Team 42 • NMAM Institute of Technology
