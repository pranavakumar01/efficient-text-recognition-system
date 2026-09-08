@echo off
REM ===========================================================================
REM  SSL ablation: does SimCLR initialisation actually improve accuracy?
REM
REM  Run this AFTER run_training.bat, because it compares against the baseline
REM  checkpoint that script produces.
REM
REM  Why this is a separate script
REM  ----------------------------
REM  The old pipeline pre-trained a SimCLR backbone and then called src.train
REM  without --use_ssl, so the SSL weights were never loaded. The project
REM  reported an SSL contribution it never measured. An ablation is the only way
REM  to make a claim about SSL: train the same recogniser twice, once from random
REM  init and once from the SimCLR backbone, and compare test CER.
REM
REM  Set expectations honestly before you spend the time:
REM    * SimCLR normally needs batches of 256-8192 to have enough negatives.
REM      At batch 32 on CPU the contrastive signal is weak.
REM    * The current SimCLR_SSL_Backbone ends in AdaptiveAvgPool2d((1,1)), which
REM      averages away the horizontal position information a CTC recogniser
REM      depends on, and it has fewer conv stages than the recogniser, so only
REM      the early layers transfer at all.
REM  A null or negative result here is a legitimate finding and belongs in the
REM  report. Do not tune until it looks positive.
REM ===========================================================================

TITLE SSL Ablation - does SimCLR init help?
cd /d "%~dp0"
cls

set PY=.\venv\Scripts\python.exe
set SSLDIR=src\models\checkpoints\ssl_init

echo ============================================================
echo   SSL ABLATION: random init  vs  SimCLR init
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
if not exist "src\models\checkpoints\cnn_bilstm_best.pth" (
    echo [X] No baseline checkpoint. Run run_training.bat first - there is
    echo     nothing to compare against yet.
    goto :failed
)

echo [1/3] SimCLR pre-training on the TRAINING SPLIT ONLY...
echo       ^(--manifest keeps test images out, so the comparison stays honest^)
%PY% -u -m src.train_ssl --manifest data\splits.csv --split train --epochs 15 --batch_size 32
if errorlevel 1 goto :failed
echo.

echo [2/3] Training the recogniser FROM the SimCLR backbone...
echo       Saved to %SSLDIR% so the baseline checkpoint is preserved.
%PY% -u -m src.train --epochs 60 --batch_size 16 --patience 12 --use_ssl --save_dir "%SSLDIR%"
if errorlevel 1 goto :failed
echo.

echo [3/3] Scoring the SSL-init model on the same held-out test split...
%PY% -u -m src.evaluate --split test --model_type cnn ^
    --checkpoint "%SSLDIR%\cnn_bilstm_best.pth" ^
    --output_csv docs\benchmark_ssl_init.csv ^
    --per_sample_csv docs\benchmark_ssl_init_per_sample.csv
if errorlevel 1 goto :failed
echo.

echo ============================================================
echo   [OK] ABLATION COMPLETE
echo ============================================================
echo.
echo   Random init CER : docs\benchmark_results.csv
echo   SimCLR init CER : docs\benchmark_ssl_init.csv
echo.
echo   Both were scored on the same test split with the same decoder, so
echo   the difference is attributable to the initialisation and nothing
echo   else. Put both rows in the report whichever way it comes out.
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   [X] ABLATION FAILED - see the error above
echo ============================================================
echo.
pause
exit /b 1
