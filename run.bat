@echo off
setlocal enabledelayedexpansion
title LocalDub - 100%% Local AI Video Dubbing and Voice Cloner

echo ========================================================
echo   LocalDub: Offline AI Video Dubber and Voice Cloner
echo   Subtitle-Driven Multi-Speaker AI Dubbing with Voice Cloning
echo ========================================================
echo.

IF NOT EXIST "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please create it first or run setup.
    pause
    exit /b 1
)

:: Check if gradio is installed
.\venv\Scripts\python.exe -c "import gradio" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Dependencies are not fully installed yet.
    echo [INFO] Installing required packages from requirements.txt...
    echo [INFO] Please wait, this may take a few minutes on first run.
    echo.
    .\venv\Scripts\pip.exe install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Package installation encountered an issue.
        pause
        exit /b 1
    )
)

echo.
echo ========================================================
echo Starting LocalDub Web UI on http://127.0.0.1:7860 ...
echo ========================================================
echo.
.\venv\Scripts\python.exe app.py
pause
