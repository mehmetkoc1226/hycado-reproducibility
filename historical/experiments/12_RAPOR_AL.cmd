@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m hycado_start.frozen_val --report-only
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo Gonderilecek dosya:
echo experiments\official_val_a025_v1_seed42\12_sabit_val_rapor.zip
pause
exit /b 0
:failed
echo Rapor olusturulamadi. Ekrandaki hata metnini gonderin.
pause
exit /b 1
:missing
echo Mevcut proje klasorunde calistirin.
pause
exit /b 2
