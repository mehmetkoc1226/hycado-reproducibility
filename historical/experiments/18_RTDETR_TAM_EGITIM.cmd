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
if not exist "hycado_start\full_image_training.py" goto missing
echo 18: GERCEK RT-DETR TAM GORUNTU EGITIMI - 80 TUR.
echo Resmi COCO'dan yeni baslangic. 17 pilot agirligi kullanilmaz.
echo 17 olcumune gore yalniz egitim hesabi yaklasik 59.4 saat; ek islemler haric.
echo Once veri hazirlanir. Diger GPU islerini kapatin; uyku modunu kapatin.
echo Kesilirse ayni CMD'yi yeniden calistirin. 12A'yi tekrarlamayin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.full_image_training
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/rtdetr_full_image_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('completed_rounds')==80 and r.get('training_samples')==465920 and r.get('optimizer_attempts')==58240 and r.get('pilot_checkpoint_used') is False else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 80 TUR TAMAMLANDI. Gonderilecek dosya:
echo experiments\rtdetr_full_image_v1_seed42\18_tam_egitim_rapor.zip
pause
exit /b 0
:failed
echo EGITIM TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.full_image_training --report-only
echo Rapor: experiments\rtdetr_full_image_v1_seed42\18_tam_egitim_rapor.zip
pause
exit /b 1
:missing
echo Paketin icindekileri mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
pause
exit /b 2
