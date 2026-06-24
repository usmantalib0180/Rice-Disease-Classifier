@echo off
setlocal

set "APP_DIR=%~dp0"
set "PYTHON=%APP_DIR%..\..\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Could not find the workspace virtual environment at "%PYTHON%".
    exit /b 1
)

"%PYTHON%" -m streamlit run "%APP_DIR%main.py"