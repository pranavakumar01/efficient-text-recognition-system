@echo off
TITLE OCR Model Training & SSL Pipeline
cls

echo ============================================================
echo   OCR Model Training & Self-Supervised Learning Pipeline
echo ============================================================
echo.

echo [1/4] Generating Synthetic Dataset Samples...
.\venv\Scripts\python.exe data\generate_synthetic.py
echo.

echo [2/4] Running SimCLR Self-Supervised Learning Pre-Training...
.\venv\Scripts\python.exe -m src.train_ssl --epochs 5 --batch_size 8
echo.

echo [3/4] Training CNN + BiLSTM + Attention OCR Model...
.\venv\Scripts\python.exe -m src.train --epochs 5 --batch_size 8
echo.

echo [4/4] Training / Evaluating Vision Transformer (TrOCR) OCR Model...
.\venv\Scripts\python.exe -m src.train_transformer --epochs 3 --batch_size 4
echo.

echo ============================================================
echo   [OK] Separated OCR Model Training Pipelines Completed!
echo   Model Checkpoints Saved to: src\models\checkpoints\
echo ============================================================
echo.
pause
