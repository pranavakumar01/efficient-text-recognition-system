@echo off
REM ===========================================================================
REM  Multi-Domain End-to-End OCR Training Pipeline (CNN + Transformer)
REM  Trains the entire dataset for both CNN-BiLSTM-Attention and Vision Transformer
REM  across Historical Documents, Mathematical LaTeX Equations, and Handwritten Texts.
REM ===========================================================================

TITLE OCR Multi-Domain Pipeline - CNN + Transformer (Historical, Math, Handwritten)
cd /d "%~dp0"
cls

set PY=.\venv\Scripts\python.exe

echo ============================================================
echo   OCR END-TO-END TRAINING PIPELINE
echo   CNN-BiLSTM-Attention + Vision Transformer (TrOCR)
echo   Domains: Historical Documents, Mathematical, Handwritten
echo ============================================================
echo.

if not exist "%PY%" (
    echo [X] Python not found at %PY%
    echo     Expected a virtual environment in this folder. Create one with:
    echo         python -m venv venv
    echo         venv\Scripts\pip install -r requirements.txt
    goto :failed
)

REM Step 1: Ensure Split Manifest Exists
if not exist "data\splits.csv" (
    echo [1/3] Generating standardized leak-free manifest...
    %PY% -u -m src.make_splits
    if errorlevel 1 goto :failed
) else (
    echo [1/3] Found existing data\splits.csv split manifest.
)
echo.

REM Step 2: Smoke test verification
echo [2/3] Smoke test: verifying both architectures on 16 samples...
%PY% -u train_all.py --max-samples 16 --cnn-epochs 1 --transformer-epochs 1 --no-evaluate
if errorlevel 1 (
    echo.
    echo [X] The smoke test failed. Check the error log above.
    goto :failed
)
echo.
echo       [OK] Verification passed! Starting full multi-domain training.
echo.

REM Step 3: Run Full Multi-Domain Training (CNN + Transformer + Quantization + Evaluation)
echo [3/3] Training both CNN and Transformer models on entire dataset...
%PY% -u train_all.py --model both --domain all --cnn-epochs 30 --transformer-epochs 5
if errorlevel 1 goto :failed
echo.

echo ============================================================
echo   [OK] MULTI-DOMAIN TRAINING COMPLETE!
echo ============================================================
echo.
echo   CNN Checkpoint       : src\models\checkpoints\cnn_bilstm_best.pth
echo   Edge INT8 Quantized  : src\models\checkpoints\cnn_bilstm_quantized.pth
echo   TrOCR Transformer    : src\models\checkpoints\trocr_finetuned\
echo   Test Benchmark CSV   : docs\benchmark_train_all.csv
echo   Full Log History     : src\models\checkpoints\train_all_history.json
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   [X] TRAINING PIPELINE FAILED - see the error above
echo ============================================================
echo.
pause
exit /b 1
