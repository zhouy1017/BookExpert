@echo off
REM BookExpert — Package to standalone EXE using PyInstaller
REM Run from project root after install.bat completes

echo ============================================
echo  BookExpert — EXE Packager (PyInstaller)
echo ============================================

cd /d "%~dp0.."

REM Install PyInstaller in the venv
echo [1/3] Installing PyInstaller...
bookexpert\Scripts\pip install pyinstaller --quiet

REM Collect all required data dirs/files
echo [2/3] Building EXE...
bookexpert\Scripts\pyinstaller ^
    --name BookExpert ^
    --onefile ^
    --noconsole ^
    --add-data "src;src" ^
    --add-data "db;db" ^
    --hidden-import streamlit ^
    --hidden-import langchain ^
    --hidden-import langchain_google_genai ^
    --hidden-import langchain_openai ^
    --hidden-import qdrant_client ^
    --hidden-import jieba ^
    --hidden-import rank_bm25 ^
    app.py

echo.
if errorlevel 1 (
    echo ERROR: Build failed. Check output above.
) else (
    echo [3/3] Done! Executable saved to dist\BookExpert.exe
    echo.
    echo NOTE: You must place google.apikey and deepseek.apikey
    echo       in the same folder as BookExpert.exe before running.
    echo.
    echo Run: dist\BookExpert.exe
)
pause
