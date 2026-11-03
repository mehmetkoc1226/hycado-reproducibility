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
if not exist "hycado_start\optimized_runtime.py" goto missing
if not exist "hycado_start\nms_equivalence.py" goto missing
if not exist "experiments\nms_equivalence_v1_seed42\result.json" goto missing
echo 15: HIZLANDIRILMIS SABIT CIKARIM VE AYRI CROP TANI.
echo Egitim yok. Batch=1, ayni agirliklar, ayni crop ve NMS kurallari.
echo Baseline ve hibrit icin 548 goruntu, 3 tur; sonra 32 goruntu tani.
echo Her yeni sonuc 12 ile birebir karsilastirilir; farkta durur.
echo Diger GPU islerini kapatin. 12A yedegini yeniden calistirmayin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.optimized_runtime
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/optimized_runtime_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('phases')==6 and r.get('images_per_phase')==548 and r.get('profile_images')==32 and r.get('all_measured_outputs_exact_original12') is True and r.get('profile_raw_queries_and_outputs_exact12') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 15 OLCUM TAMAMLANDI. Gonderilecek dosya:
echo experiments\optimized_runtime_v1_seed42\15_hizlandirilmis_rapor.zip
pause
exit /b 0
:failed
echo OLCUM TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.optimized_runtime --report-only
echo Varsa guncel raporu gonderin:
echo experiments\optimized_runtime_v1_seed42\15_hizlandirilmis_rapor.zip
pause
exit /b 1
:missing
echo Paketin icindekileri D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 12, 13, 14 kaynaklari ve onceki moduller gerekli.
pause
exit /b 2
