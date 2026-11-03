@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONHASHSEED=42
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\score_ablation.py" goto missing
if not exist "experiments\raw_crop_capture_v1_seed42\capture_index.json" goto missing
echo 11D: CPU ILE AYRI GELISTIRME SKOR DENEYI.
echo Egitim veya GPU cikarimi yok. 11 ve 11C kayitlari korunur.
echo Crop skor carpanlari: 1, 0.75, 0.50, 0.25, 0.10.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.score_ablation
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/score_ablation_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('images')==647 and r.get('crops')==10682 and r.get('variants')==7 and r.get('baseline_and_alpha1_original11_metrics_equal') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo YEDI VARYANT TAMAMLANDI. Gonderilecek dosya:
echo experiments\score_ablation_v1_seed42\11D_skor_rapor.zip
pause
exit /b 0
:failed
echo DENEY TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.score_ablation --report-only
echo Varsa su guncel raporu gonderin:
echo experiments\score_ablation_v1_seed42\11D_skor_rapor.zip
pause
exit /b 1
:missing
echo Paketin icerigini mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 11 ve 11C klasorleri, ham NPZ dosyalariyla birlikte gerekli.
pause
exit /b 2
