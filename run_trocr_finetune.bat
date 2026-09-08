@echo off
REM ===========================================================================
REM  TrOCR fine-tuning. Separate from run_training.bat because it is expensive
REM  and because it is the model that should carry the mathematics.
REM
REM  Why fine-tuning is not optional for TrOCR here
REM  ---------------------------------------------
REM  microsoft/trocr-base-printed has never seen LaTeX. It cannot emit
REM  \frac{dy}{dt} no matter how wide the beam, so its CER on the mathwriting
REM  subset is bounded near 1.0 out of the box - that is a property of the
REM  checkpoint, not a bug to decode around. Fine-tuning is how an
REM  encoder-decoder learns a new output alphabet.
REM
REM  And unlike the CTC recogniser, TrOCR has no monotonic alignment constraint,
REM  so 2D math layout is representable for it. That is why math is INCLUDED
REM  here and EXCLUDED for the CNN.
REM
REM  Cost
REM  ----
REM  trocr-base-* is ~334M parameters. On CPU one epoch over the training split
REM  takes hours. This script uses trocr-small-printed (~62M) by default so the
REM  run finishes; pass base explicitly if you have a GPU.
REM
REM  Step 1 is a 40-sample smoke test. The previous train_transformer.py never
REM  performed a backward pass at all, so before trusting a long run, confirm
REM  that the loss actually moves.
REM ===========================================================================

TITLE TrOCR Fine-Tuning
cd /d "%~dp0"
cls

set PY=.\venv\Scripts\python.exe
set MODEL=microsoft/trocr-small-printed

echo ============================================================
echo   TrOCR FINE-TUNING  (model: %MODEL%)
echo ============================================================
echo.

if not exist "%PY%" (
    echo [X] Python not found at %PY%
    goto :failed
)
if not exist "data\splits.csv" (
    echo [X] data\splits.csv not found. Run run_training.bat first.
    goto :failed
)

echo [1/3] Smoke test: 40 samples, 1 epoch. Checking the loss moves at all...
%PY% -u -m src.train_transformer --model %MODEL% --max-samples 40 --epochs 1 --batch_size 2
if errorlevel 1 (
    echo.
    echo [X] Smoke test failed. Stopping before the long run.
    goto :failed
)
echo.
echo       [OK] Backward pass works. Starting the real fine-tune.
echo.

echo [2/3] Fine-tuning on the training split ^(math INCLUDED^)...
echo       Zero-shot CER is measured first, so the report can state the
echo       before/after rather than a single unanchored number.
echo       This is slow. Leave it running.
%PY% -u -m src.train_transformer --model %MODEL% --epochs 8 --batch_size 4 --lr 5e-5
if errorlevel 1 goto :failed
echo.

echo [3/3] Scoring the fine-tuned model on the held-out test split...
%PY% -u -m src.evaluate --split test --model_type transformer ^
    --trocr-model src\models\checkpoints\trocr_finetuned ^
    --output_csv docs\benchmark_trocr_finetuned.csv ^
    --per_sample_csv docs\benchmark_trocr_finetuned_per_sample.csv
if errorlevel 1 goto :failed
echo.

echo ============================================================
echo   [OK] TrOCR FINE-TUNING COMPLETE
echo ============================================================
echo.
echo   Weights   : src\models\checkpoints\trocr_finetuned\
echo   History   : src\models\checkpoints\trocr_training_history.json
echo               ^(epoch 0 is the zero-shot baseline^)
echo   Test CER  : docs\benchmark_trocr_finetuned.csv
echo   Per-sample: docs\benchmark_trocr_finetuned_per_sample.csv
echo.
echo   For handwritten data specifically, try:
echo     set MODEL=microsoft/trocr-base-handwritten
echo   and run this script again.
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   [X] TrOCR FINE-TUNING FAILED - see the error above
echo ============================================================
echo.
pause
exit /b 1
