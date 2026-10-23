@echo off
setlocal
set "KEIBA_REPARSE_PYTHON=%USERPROFILE%\keiba\.venv\Scripts\python.exe"
if not exist "%KEIBA_REPARSE_PYTHON%" (
    echo Python was not found in the keiba virtual environment.
    pause
    exit /b 1
)
"%KEIBA_REPARSE_PYTHON%" -X utf8 "%~dp0reparse_saved_data.py" %*
set "KEIBA_REPARSE_EXIT=%ERRORLEVEL%"
pause
exit /b %KEIBA_REPARSE_EXIT%
