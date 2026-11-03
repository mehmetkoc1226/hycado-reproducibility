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
if not exist "hycado_start\full_image_pilot.py" goto missing
if not exist "experiments\size_diagnostics_v1_seed42\result.json" goto missing
echo 17: BAGIMSIZ RT-DETR TAM GORUNTU KISA EGITIM.
echo Resmi COCO baslangic agirligi. Ayni 5824/647 TRAIN ayrimi.
echo 960 piksel, fiziksel batch=1, etkin batch=8; toplam 768 egitim ornegi.
echo Bu komut 80 turluk uzun egitimi BASLATMAZ.
echo En yogun 16 TRAIN goruntusu de denenir. Diger GPU islerini kapatin.
echo Veri ve checkpoint D: uzerinde yeni 17 klasorune yazilir. 12A'yi tekrarlamayin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.full_image_pilot
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/rtdetr_full_image_pilot_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('phases')==3 and r.get('training_samples')==768 and r.get('optimizer_attempts')==96 and r.get('amp_quality',{}).get('overall_update_fraction',0)>=0.90 and r.get('official_val_rows_used_for_training') is False and r.get('long_training_started') is False else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 17 KISA EGITIM TAMAMLANDI. Gonderilecek dosya:
echo experiments\rtdetr_full_image_pilot_v1_seed42\17_tam_goruntu_rapor.zip
pause
exit /b 0
:failed
echo KISA EGITIM TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.full_image_pilot --report-only
echo Mevcut raporu gonderin:
echo experiments\rtdetr_full_image_pilot_v1_seed42\17_tam_goruntu_rapor.zip
pause
exit /b 1
:missing
echo Paketin icindekileri mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Onceki kaynaklar, kurulum, veri ve tamamlanan 16 raporu gerekli.
pause
exit /b 2
