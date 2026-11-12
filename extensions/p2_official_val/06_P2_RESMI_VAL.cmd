@echo off
setlocal
set "ROOT=D:\HyCADO_Temiz_Baslangic"
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
 echo Beklenen Python bulunamadi: "%PY%"
 pause
 exit /b 1
)
"%PY%" "%~dp0p2_official_val.py" --project "%ROOT%"
if errorlevel 1 (
 echo Resmi VAL durdu. Rapor: "%ROOT%\experiments\p2_strong_baselines_v1_seed42\06_p2_resmi_val_rapor.zip"
 pause
 exit /b 1
)
echo Gonderilecek rapor: "%ROOT%\experiments\p2_strong_baselines_v1_seed42\06_p2_resmi_val_rapor.zip"
pause
