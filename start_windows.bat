@echo off
REM start_windows.bat — launch TripoSR server on Windows
set PYTHONPATH=%~dp0TripoSR;%PYTHONPATH%
call .venv\Scripts\activate.bat 2>nul || (
    echo X Run setup_windows.bat first.
    pause & exit /b 1
)
echo.
echo   Starting TripoSR server on http://localhost:7860
echo   Open index.html in your browser
echo   First run downloads ~2 GB model weights
echo   Press Ctrl+C to stop
echo.
python app.py
