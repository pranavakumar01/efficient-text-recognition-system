@echo off
TITLE Major Project Code 42 - Web Demo Launcher
COLOR 0A
echo =======================================================================
echo     MAJOR PROJECT CODE 42: EFFICIENT TEXT RECOGNITION (DL + SSL)
echo =======================================================================
echo.
echo [1/3] Activating Virtual Environment...
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo [2/3] Generating Evaluation Sample Images...
python data\generate_samples.py

echo [3/3] Starting FastAPI Web Server at http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload

pause
