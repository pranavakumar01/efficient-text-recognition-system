@echo off
TITLE Major Project Code 42 - Online Demo Launcher (Public HTTPS)
COLOR 0B
echo =======================================================================
echo     MAJOR PROJECT CODE 42: EFFICIENT TEXT RECOGNITION (DL + SSL)
echo             TEMPORARY PUBLIC ONLINE DEMO LAUNCHER
echo =======================================================================
echo.
echo [1/3] Activating Virtual Environment...
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo [2/3] Verifying Evaluation Sample Images...
python data\generate_samples.py

echo [3/3] Starting Web Server + Public HTTPS Cloudflare Tunnel ...
python run.py --server --port 8000 --tunnel

pause
