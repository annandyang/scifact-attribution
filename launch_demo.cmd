@echo off
setlocal
cd /d "%~dp0"
set "SCIFACT_PY=.venv\Scripts\python.exe"
if not exist "%SCIFACT_PY%" (
    echo Set up Python 3.11 and requirements-demo.txt first. See docs\QUICKSTART.zh.md.
    pause
    exit /b 1
)
set GRADIO_ANALYTICS_ENABLED=False
"%SCIFACT_PY%" app\gradio_app.py --split dev --port 7860 --open-browser
if errorlevel 1 pause
