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
if not exist "hycado_start\runtime_benchmark.py" goto missing
if not exist "experiments\official_val_a025_v1_seed42\result.json" goto missing
echo 13: SABIT BASELINE / HIBRIT HIZ VE GPU BELLEGI.
echo Egitim yok. 548 goruntu, her kosulda 3 olcum turu.
echo Eski tahminler sure yerine kullanilmaz; gercek GPU cikarimi yapilir.
echo Diger GPU islerini kapatin. 12A yedegini yeniden calistirmayin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.runtime_benchmark
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/runtime_memory_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('phases')==6 and r.get('images_per_phase')==548 and r.get('all_measured_outputs_exact_original12') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo HIZ VE GPU BELLEK OLCUMU TAMAMLANDI.
echo Gonderilecek dosya:
echo experiments\runtime_memory_v1_seed42\13_hiz_bellek_rapor.zip
pause
exit /b 0
:failed
echo OLCUM TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.runtime_benchmark --report-only
echo Varsa guncel raporu gonderin:
echo experiments\runtime_memory_v1_seed42\13_hiz_bellek_rapor.zip
pause
exit /b 1
:missing
echo Paketi mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 12 kaynaklari, veriler ve agirliklar gerekli.
pause
exit /b 2
