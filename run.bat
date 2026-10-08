@echo off
setlocal enabledelayedexpansion
title LocalDub - 100%% Local AI Video Dubbing and Voice Cloner

echo ========================================================
echo   LocalDub: Professional AI Video Dubber & Voice Cloner
echo   Subtitle-Driven Multi-Speaker AI Dubbing Workstation
echo ========================================================
echo.

:: 1. Check if virtual environment exists; if not, create it automatically
IF NOT EXIST "venv\Scripts\python.exe" (
    echo [SETUP] First-time setup detected. Virtual environment 'venv' not found.
    echo [SETUP] Creating Python virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.10-3.12 is installed and added to PATH.
        pause
        exit /b 1
    )
    echo [SETUP] Virtual environment created successfully.
    echo.
    echo [SETUP] Installing PyTorch with CUDA 12.4 acceleration...
    .\venv\Scripts\python.exe -m pip install --upgrade pip
    .\venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo.
    echo [SETUP] Installing requirements from requirements.txt...
    .\venv\Scripts\pip.exe install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation encountered an issue.
        pause
        exit /b 1
    )
    echo.
    echo [SETUP] Downloading verified AI models from official Hugging Face sources...
    .\venv\Scripts\python.exe download_models.py
)

:: 2. Verify dependencies are present
.\venv\Scripts\python.exe -c "import gradio, torch" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Updating dependencies from requirements.txt...
    .\venv\Scripts\pip.exe install -r requirements.txt
)

:: 3. Launch LocalDub Workstation
echo.
echo ========================================================
echo   Starting LocalDub Web UI on http://127.0.0.1:7860 ...
echo ========================================================
echo.
.\venv\Scripts\python.exe app.py
pause
