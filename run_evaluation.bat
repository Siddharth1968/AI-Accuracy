@echo off
setlocal enabledelayedexpansion
title VibeShift AI Accuracy Evaluator
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Local Python environment .venv not found.
    echo         Please run setup.bat first.
    exit /b 1
)

:: If arguments were provided on command line, pass directly to evaluate.py
if not "%~1"=="" (
    ".venv\Scripts\python.exe" -B evaluate.py %*
    exit /b %ERRORLEVEL%
)

echo ========================================================
echo  VibeShift AI Accuracy Evaluator - Console Runner
echo ========================================================
echo.
".venv\Scripts\python.exe" -B evaluate.py --list-datasets
echo.
set /p "DS_NAME=Enter dataset name [e.g. FER2013, ExpW_Eval1400, JAFFE, CREMA_D_Frames144, dataset]: "
if "%DS_NAME%"=="" (
    echo [CANCELLED] No dataset entered.
    exit /b 0
)

set /p "SP_NAME=Enter split name [leave blank for single-split datasets; e.g. PrivateTest / full_photo / annotated_crops]: "
set /p "INF_MODE=Select mode [single / flip / both, default=single]: "
if "%INF_MODE%"=="" set "INF_MODE=single"

set /p "PROTO=Select protocol [auto / pre_cropped / full_photo, default=auto]: "
if "%PROTO%"=="" set "PROTO=auto"

echo.
if "%SP_NAME%"=="" (
    ".venv\Scripts\python.exe" -B evaluate.py --dataset "%DS_NAME%" --mode "%INF_MODE%" --protocol "%PROTO%"
) else (
    ".venv\Scripts\python.exe" -B evaluate.py --dataset "%DS_NAME%" --split "%SP_NAME%" --mode "%INF_MODE%" --protocol "%PROTO%"
)
set "RUN_RC=%ERRORLEVEL%"
echo.
pause
exit /b %RUN_RC%
endlocal
