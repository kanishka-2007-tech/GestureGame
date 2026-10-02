@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found on PATH. Install Python 3.10+ and retry.
  pause
  exit /b 1
)

python -c "import cv2, mediapipe, websockets, numpy" >nul 2>&1
if errorlevel 1 (
  echo Installing gesture control dependencies...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
  )
)

echo.
echo   Gesture Subway Runner
echo   1. Keep your webcam clear and well lit.
echo   2. The game page opens automatically - do not close this window.
echo.
python gesture_server.py %*
if errorlevel 1 pause
endlocal