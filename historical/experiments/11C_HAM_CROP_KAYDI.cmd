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
if not exist "hycado_start\crop_capture.py" goto missing
if not exist "hycado_start\dev_diagnostics.py" goto missing
if not exist "experiments\development_eval_v1_seed42\result.json" goto missing
echo HAM CROP KAYDI: Ayni 80. tur EMA, 647 goruntu ve 10682 crop.
echo Egitim veya esik aramasi yapilmaz. GPU'da bir cikarim gecisi yapilir.
echo Diger GPU islerini kapatin. En az 3 GiB bos disk alani gerekli.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.crop_capture
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/raw_crop_capture_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('images')==647 and r.get('crops')==10682 else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 647 GORUNTUNUN HAM CROP VE NMS IZ KAYDI TAMAMLANDI.
echo Gonderilecek dosya:
echo experiments\raw_crop_capture_v1_seed42\11C_ham_crop_rapor.zip
pause
exit /b 0
:failed
echo KAYIT TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.crop_capture --report-only
echo Varsa su guncel raporu gonderin:
echo experiments\raw_crop_capture_v1_seed42\11C_ham_crop_rapor.zip
pause
exit /b 1
:missing
echo Paketin icerigini mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 11 verisi ve 11B kaynak modulu de gerekli.
pause
exit /b 2
