@echo off
REM ===========================================================================
REM  Main training pipeline: CNN + BiLSTM + Attention with CTC.
REM
REM  What was wrong with the previous version of this file
REM  ----------------------------------------------------
REM  1. It never called `src.make_splits`, so there was no train/valid/test
REM     manifest. Training and evaluation both ran over the whole data
REM     directory, which is why every reported CER was measured on data the
REM     model had trained on.
REM  2. It ran `src.train_ssl` and then `src.train` WITHOUT --use_ssl, so the
REM     SimCLR backbone was pretrained and then thrown away. The SSL stage of
REM     the project had no effect on any reported result.
REM  3. It generated synthetic data AFTER nothing and before the manifest did
REM     not exist, so ordering was never enforced.
REM  4. No step checked its exit code. A failure in step 1 carried on to step 4
REM     and surfaced as a confusing unrelated error much later.
REM  5. `--epochs 5` on a from-scratch CTC model is not enough for the loss to
REM     leave the blank-collapse plateau.
REM  6. No `cd` to the script directory, so `.\venv\Scripts\python.exe` failed
REM     whenever the script was launched from another working directory.
REM
REM  TrOCR fine-tuning and the SSL ablation are deliberately NOT here - they
REM  are expensive and optional. See run_trocr_finetune.bat and
REM  run_ssl_ablation.bat.
REM ===========================================================================

TITLE OCR Training Pipeline - CNN + BiLSTM + Attention (CTC)
cd /d "%~dp0"
cls

set PY=.\venv\Scripts\python.exe

echo ============================================================
echo   OCR TRAINING PIPELINE
echo   CNN + BiLSTM + Attention, CTC loss, leak-free splits
echo ============================================================
echo.

if not exist "%PY%" (
    echo [X] Python not found at %PY%
    echo     Expected a virtual environment in this folder. Create one with:
    echo         python -m venv venv
    echo         venv\Scripts\pip install -r requirements.txt
    goto :failed
)

echo [1/6] Generating synthetic samples...
echo       ^(must run before the manifest, so the new files are indexed^)
%PY% -u data\generate_synthetic.py
if errorlevel 1 goto :failed
echo.

echo [2/6] Building the leak-free split manifest...
echo       ^(splits by distinct label string, so no line appears in two splits^)
%PY% -u -m src.make_splits
if errorlevel 1 goto :failed
echo.

echo [3/6] Smoke test: 1 epoch on 32 samples to prove the loop runs...
echo       ^(writes cnn_bilstm_smoketest.pth - the real checkpoint is untouched^)
%PY% -u -m src.train --epochs 1 --batch_size 8 --max-samples 32 --warmup-steps 5
if errorlevel 1 (
    echo.
    echo [X] The smoke test failed. Stopping here on purpose: there is no point
    echo     spending hours on a full run when one epoch cannot complete.
    echo     Send me the error above.
    goto :failed
)
echo.
echo       [OK] Pipeline runs end to end. Starting the real training.
echo.

echo [4/6] Training the recogniser...
echo       Early stopping on validation CER, patience 12.
echo       This is the long step - budget a couple of hours on CPU.
echo       Text lines only by default; LaTeX math is excluded because CTC
echo       cannot represent 2D layout ^(add --include-math to override^).
%PY% -u -m src.train --epochs 60 --batch_size 16 --patience 12
if errorlevel 1 goto :failed
echo.

echo [5/6] Evaluating on the HELD-OUT TEST split...
echo       Raw decode, no autocorrect, no fallback engine. This is the first
echo       number in this project that means anything.
%PY% -u -m src.evaluate --split test --model_type cnn
if errorlevel 1 goto :failed
echo.

echo [6/6] Regenerating report figures from the measured results...
%PY% -u -m src.utils.generate_charts
if errorlevel 1 goto :failed
echo.

echo ============================================================
echo   [OK] PIPELINE COMPLETE
echo ============================================================
echo.
echo   Checkpoint      : src\models\checkpoints\cnn_bilstm_best.pth
echo   Training curve  : src\models\checkpoints\training_history.json
echo   Test summary    : docs\benchmark_results.csv
echo   Per-sample I/O  : docs\benchmark_per_sample.csv    ^<-- READ THIS ONE
echo   Figures         : docs\figures\
echo.
echo   Open benchmark_per_sample.csv and read the ground_truth and
echo   prediction columns side by side. The aggregate CER tells you how
echo   much is wrong; only that file tells you WHAT is wrong.
echo.
echo   Optional next steps:
echo     run_ssl_ablation.bat     - does SimCLR init actually help?
echo     run_trocr_finetune.bat   - fine-tune TrOCR ^(slow, needed for math^)
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   [X] PIPELINE FAILED - see the error above
echo ============================================================
echo   Nothing downstream was run, so no results were overwritten
echo   with output from a broken stage.
echo.
pause
exit /b 1
