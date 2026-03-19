@echo off
REM BookExpert — One-Click Launcher

cd /d "%~dp0.."

if not exist "bookexpert\Scripts\python.exe" (
    echo Not installed yet. Please run release\install.bat first.
    pause & exit /b 1
)

echo Starting BookExpert...
echo Open http://localhost:8501 in your browser.
echo Press Ctrl+C to stop.
echo.

bookexpert\Scripts\streamlit run app.py --server.port 8501
