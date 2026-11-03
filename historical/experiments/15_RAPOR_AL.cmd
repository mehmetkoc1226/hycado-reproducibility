@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m hycado_start.optimized_runtime --report-only
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo Rapor: experiments\optimized_runtime_v1_seed42\15_hizlandirilmis_rapor.zip
pause
exit /b 0
:failed
echo Rapor olusturulamadi. Cikis: %HYCADO_EXIT%
pause
exit /b 1
:missing
echo Mevcut proje klasorunden calistirin.
pause
exit /b 2
