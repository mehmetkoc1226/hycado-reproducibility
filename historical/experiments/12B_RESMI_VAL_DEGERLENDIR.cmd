@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONHASHSEED=42
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\frozen_val.py" goto missing
if not exist "reports\12_yedek_sonuc.json" goto backup
echo 12B: 548 RESMI VAL GORUNTUSUNDE SABIT BASELINE / HIBRIT KARSILASTIRMASI.
echo Egitim yok. Crop skoru x0.25 sabit. Esik aramasi yok.
echo Once yedek/kod/model/veri kimlikleri dogrulanacak.
echo Diger GPU egitimlerini kapatin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.frozen_val
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/official_val_a025_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('images')==548 and r.get('crop_score_multiplier')==0.25 and r.get('backup_verified') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 548 GORUNTUNUN SABIT VAL DEGERLENDIRMESI TAMAMLANDI.
echo Gonderilecek dosya:
echo experiments\official_val_a025_v1_seed42\12_sabit_val_rapor.zip
pause
exit /b 0
:failed
echo DEGERLENDIRME TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.frozen_val --report-only
echo Varsa guncel raporu gonderin:
echo experiments\official_val_a025_v1_seed42\12_sabit_val_rapor.zip
pause
exit /b 1
:backup
echo Once 12A_YEDEK_AL.cmd komutunu basariyla tamamlayin.
pause
exit /b 2
:missing
echo Paketi mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
pause
exit /b 2
