@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Velum 4.10 - Standalone Web Frontend
echo ============================================
set "PY=python"
set "CPU_FLAG="
if exist "%~dp0runtime\python.exe" (
  set "PY=%~dp0runtime\python.exe"
  set "CPU_FLAG=--cpu"
  goto launch
)
if exist "%~dp0..\..\..\python\python.exe" set "PY=%~dp0..\..\..\python\python.exe"
if exist "%~dp0..\..\..\python_embeded\python.exe" set "PY=%~dp0..\..\..\python_embeded\python.exe"
if exist "%~dp0..\..\.venv\Scripts\python.exe" set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if defined VELUM_PYTHON set "PY=%VELUM_PYTHON%"
:launch
echo Close this window to stop the standalone webpage.
"%PY%" -I -B -u "%~dp0launch_web.py" %CPU_FLAG% %*
if errorlevel 1 (
  echo [ERROR] See the Python path, failed dependency and diagnostic log above.
  echo If Python is missing, get the full CPU portable ZIP, not the source-only plugin.
  pause
  exit /b 1
)
endlocal
