@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "experiments\rtdetr_crop_v1_seed42\protocol.json" goto missing
".venv\Scripts\python.exe" -m hycado_start.crop_experiment --report-only
if errorlevel 1 goto failed
echo Gonderilecek dosya:
echo experiments\rtdetr_crop_v1_seed42\10_rtdetr_crop_egitim_rapor.zip
echo Bu islem egitimi durdurmaz veya yeni egitim baslatmaz.
pause
exit /b 0
:missing
echo 10 egitimine ait protokol veya mevcut Python ortami bulunamadi.
echo Egitim basladiktan sonra bu dosyayla ara rapor alinabilir.
pause
exit /b 1
:failed
echo Rapor olusturulamadi. Terminal hata metnini paylasin.
pause
exit /b 1
