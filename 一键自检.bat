@echo off
setlocal
call "%~dp0一键启动.bat" --self-test
if errorlevel 1 exit /b 1
pause
endlocal
