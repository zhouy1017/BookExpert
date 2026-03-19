@echo off
REM BookExpert — One-Click Installer
REM Requires Python 3.10+ on PATH

echo ============================================
echo  BookExpert Installer
echo ============================================

REM Move to project root (parent of release\)
cd /d "%~dp0.."

REM Create virtual environment if missing
if not exist "bookexpert\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv bookexpert
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10+ and retry.
        pause & exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists — skipping.
)

REM Install / upgrade dependencies
echo [2/3] Installing dependencies (this may take several minutes)...
bookexpert\Scripts\pip install --upgrade pip --quiet
bookexpert\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Dependency install failed. Check requirements.txt.
    pause & exit /b 1
)

REM Check API key files
echo [3/3] Checking API key files...
if not exist "google.apikey" (
    echo.
    echo  WARNING: google.apikey not found.
    echo  Create google.apikey in the project root with your Google AI Studio key.
)
if not exist "deepseek.apikey" (
    echo.
    echo  WARNING: deepseek.apikey not found.
    echo  Create deepseek.apikey in the project root with your DeepSeek API key.
)

echo.
echo ============================================
echo  Installation complete!
echo  Run:  release\run.bat   to start the app
echo ============================================
pause
