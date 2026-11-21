@echo off
setlocal enabledelayedexpansion
title VibeShift AI Accuracy Evaluator - Setup
cd /d "%~dp0"

echo ========================================================
echo  VibeShift AI Accuracy Evaluator - Automated Setup
echo  Single Model: CleanBTNVit Phase 12 + Optional YuNet
echo ========================================================
echo.

:: 1. Verify 64-bit Windows
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
if /I not "%ARCH%"=="AMD64" if /I not "%ARCH%"=="ARM64" (
    echo [ERROR] 64-bit Windows is required. Detected: %ARCH%
    exit /b 1
)
echo [OK] Windows %ARCH% detected.

:: 2. Detect Python 3.12 / 3.11 / 3.10
set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.12"
    if not defined PYTHON_CMD (
        py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
    )
    if not defined PYTHON_CMD (
        py -3.10 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.10"
    )
)
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python scripts\env_check.py --check-python >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)
if not defined PYTHON_CMD (
    echo [ERROR] Python 3.10, 3.11, or 3.12 64-bit is required.
    echo         Download Python 3.12 from: https://www.python.org/downloads/release/python-31210/
    exit /b 1
)
for /f "usebackq delims=" %%V in (`%PYTHON_CMD% scripts\env_check.py --check-python`) do set "PY_VER=%%V"
echo [OK] Using Python %PY_VER% [%PYTHON_CMD%].

:: 3. Create isolated local virtual environment
echo.
echo [1/4] Creating local Python virtual environment [.venv]...
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul

:: 4. Detect NVIDIA CUDA driver 12.1+ or fall back to CPU PyTorch
echo [2/4] Installing PyTorch [auto NVIDIA CUDA detection with CPU fallback]...
set "USE_CUDA=0"
if /I "%VIBESHIFT_FORCE_CPU%"=="1" (
    echo [INFO] VIBESHIFT_FORCE_CPU=1 set - installing CPU PyTorch build.
) else (
    for /f "usebackq delims=" %%G in (`".venv\Scripts\python.exe" scripts\env_check.py --check-cuda`) do (
        if "%%G"=="CUDA_OK" set "USE_CUDA=1"
    )
)

".venv\Scripts\python.exe" -c "import torch" >nul 2>nul
if errorlevel 1 (
    if "%USE_CUDA%"=="1" (
        echo [INFO] Compatible NVIDIA GPU detected. Installing PyTorch CUDA 12.6...
        ".venv\Scripts\python.exe" -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
        if errorlevel 1 (
            echo [WARNING] CUDA PyTorch install failed; falling back to CPU PyTorch...
            ".venv\Scripts\python.exe" -m pip install "torch>=2.5.0"
            if errorlevel 1 (
                echo [ERROR] PyTorch installation failed.
                exit /b 1
            )
        )
    ) else (
        echo [INFO] No compatible NVIDIA CUDA GPU detected. Installing CPU PyTorch build...
        ".venv\Scripts\python.exe" -m pip install "torch>=2.5.0"
        if errorlevel 1 (
            echo [ERROR] PyTorch CPU installation failed.
            exit /b 1
        )
    )
) else (
    echo [OK] PyTorch already installed in .venv.
)

:: 5. Install minimal evaluator requirements
echo [3/4] Installing minimal Python evaluator packages [requirements.txt]...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.txt.
    exit /b 1
)

:: 6. Verify model SHA-256, YuNet SHA-256, imports, and dataset registry
echo [4/4] Verifying model assets, imports, and dataset registry...
".venv\Scripts\python.exe" -B evaluate.py --check-env
set "CHECK_RC=%ERRORLEVEL%"

echo.
if "%CHECK_RC%"=="0" (
    echo ========================================================
    echo  SUCCESS: VibeShift AI Accuracy Evaluator is READY!
    echo ========================================================
    echo  Next steps:
    echo    1. List datasets:
    echo         .venv\Scripts\python.exe evaluate.py --list-datasets
    echo    2. Run interactive console launcher:
    echo         run_evaluation.bat
    echo    3. Or run a specific dataset directly, e.g.:
    echo         .venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode flip
    echo.
    echo  In VS Code: Open this folder and select .venv\Scripts\python.exe
    echo ========================================================
    exit /b 0
)

echo ========================================================
echo  PARTIAL / CHECK FAILED [code %CHECK_RC%].
echo  If model\clean_btn_vit_phase12.pt is missing on a fresh
echo  teammate machine, see FRIEND_SETUP_GUIDE.md for authorized setup.
echo ========================================================
exit /b %CHECK_RC%
endlocal
