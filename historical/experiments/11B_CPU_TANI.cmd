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
if not exist "hycado_start\dev_diagnostics.py" goto missing
if not exist "experiments\development_eval_v1_seed42\result.json" goto missing
echo CPU TANI: mevcut tahminler incelenir. GPU veya MATLAB gerekmez.
echo Egitim, model cikarimi ve parametre aramasi yapilmaz.
".venv\Scripts\python.exe" -u -m hycado_start.dev_diagnostics
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/development_diagnostics_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('images')==647 else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo 647 GORUNTUNUN CPU TANI ANALIZI TAMAMLANDI.
echo Gonderilecek dosya:
echo experiments\development_diagnostics_v1_seed42\11B_tani_rapor.zip
pause
exit /b 0
:failed
echo CPU tani tamamlanamadi. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.dev_diagnostics --report-only
echo Varsa su raporu gonderin:
echo experiments\development_diagnostics_v1_seed42\11B_tani_rapor.zip
pause
exit /b 1
:missing
echo Paketin icerigini mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 11 degerlendirmesinin tahmin ve annotation dosyalari gerekli.
pause
exit /b 2
