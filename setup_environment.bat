@echo off
TITLE Major Project Code 42 - Setup Environment
COLOR 0B
echo =======================================================================
echo     MAJOR PROJECT CODE 42: ENVIRONMENT SETUP & DEPENDENCY INSTALLER
echo =======================================================================
echo.
cd /d "%~dp0"

echo [1/3] Creating Python Virtual Environment (venv)...
python -m venv venv

echo [2/3] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [3/3] Installing Core Packages (PyTorch, OpenCV, Transformers, FastAPI)...
python -m pip install --upgrade pip setuptools wheel
python -m pip install opencv-python pillow fastapi uvicorn python-multipart torch torchvision jiwer scikit-learn matplotlib

echo.
echo =======================================================================
echo     SETUP COMPLETE! You can now run "run_demo.bat" to start the project.
echo =======================================================================
echo.
pause
