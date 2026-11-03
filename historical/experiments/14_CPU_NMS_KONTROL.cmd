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
if not exist "hycado_start\nms_equivalence.py" goto missing
if not exist "experiments\runtime_memory_v1_seed42\result.json" goto missing
echo 14: CPU NMS ESLIK VE HIZ KONTROLU.
echo Egitim veya GPU cikarimi yok. Ayni 12 ham kayitlari kullanilir.
echo Esikler ve eski kodlar degistirilmez. 548 goruntu, uc tur.
echo Agir CPU islerini kapatin. 12A yedegini yeniden calistirmayin.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.nms_equivalence
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('experiments/nms_equivalence_v1_seed42/result.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('images_per_trial')==548 and r.get('trials')==3 and r.get('all_outputs_exact12') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo CPU NMS KARSILASTIRMASI TAMAMLANDI. Eski kod otomatik degistirilmedi.
echo Gonderilecek dosya:
echo experiments\nms_equivalence_v1_seed42\14_nms_rapor.zip
pause
exit /b 0
:failed
echo KONTROL TAMAMLANMADI. Cikis kodu: %HYCADO_EXIT%
".venv\Scripts\python.exe" -m hycado_start.nms_equivalence --report-only
echo Varsa guncel raporu gonderin:
echo experiments\nms_equivalence_v1_seed42\14_nms_rapor.zip
pause
exit /b 1
:missing
echo Paketi mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Tamamlanan 12 ham kayitlari ve 13 paketi gerekli.
pause
exit /b 2
