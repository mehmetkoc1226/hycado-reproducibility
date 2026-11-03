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
if not exist "hycado_start\dev_evaluation.py" goto missing
echo GERCEK GELISTIRME DEGERLENDIRMESI: 647 goruntu, 10682 crop.
echo Egitim yapilmaz; mevcut model agirliklari degistirilmez.
echo Resmi VAL kullanilmaz. Diger GPU islerini kapatin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.dev_evaluation
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/development_eval_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('development_images')==647 and r.get('crops')==10682 else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 647 GORUNTUNUN GELISTIRME DEGERLENDIRMESI TAMAMLANDI.
echo Python gelistirme metrikleri; resmi MATLAB esdegerlik kontrolu ayridir.
echo Gonderilecek dosya:
echo experiments\development_eval_v1_seed42\11_gelistirme_rapor.zip
pause
exit /b 0
:failed
echo DEGERLENDIRME TAMAMLANMADI. Egitim agirliklari korunur.
echo Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.dev_evaluation --report-only
echo Varsa su guncel dosyayi gonderin:
echo experiments\development_eval_v1_seed42\11_gelistirme_rapor.zip
pause
exit /b 1
:missing
echo ZIP icerigini mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
pause
exit /b 2
