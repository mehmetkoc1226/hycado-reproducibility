@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\size_diagnostics.py" goto missing
if not exist "hycado_start\optimized_runtime.py" goto missing
if not exist "experiments\optimized_runtime_v1_seed42\result.json" goto missing
echo 16: NESNE BOYUTUNA GORE SABIT VAL YAKALAMA ANALIZI.
echo CPU ile mevcut 548 goruntunun tahminleri incelenir.
echo Egitim, GPU cikarimi ve parametre aramasi yok.
echo Yeni kod ve kucuk rapor yedegi otomatik alinir. 12A'yi tekrarlamayin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.size_diagnostics
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/size_diagnostics_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('images')==548 and r.get('thresholds')==[0.5,0.75] and r.get('original12_TP_FP_counts_equal') is True and r.get('source_predictions_unchanged') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 16 ANALIZ TAMAMLANDI. Gonderilecek dosya:
echo experiments\size_diagnostics_v1_seed42\16_boyut_rapor.zip
pause
exit /b 0
:failed
echo ANALIZ TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.size_diagnostics --report-only
echo Varsa guncel raporu gonderin:
echo experiments\size_diagnostics_v1_seed42\16_boyut_rapor.zip
pause
exit /b 1
:missing
echo Paketin icindekileri mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 12, 13, 14, 15 kaynaklari ve onceki moduller gerekli.
pause
exit /b 2
