@echo off
setlocal
cd /d "%~dp0"
set "HYCADO_PROJECT=%~1"
if not defined HYCADO_PROJECT set "HYCADO_PROJECT=D:\HyCADO_Temiz_Baslangic"
set "HYCADO_PY_SELECTOR="
set "HYCADO_PY_EXE=%HYCADO_PROJECT%\.venv\Scripts\python.exe"
if exist "%HYCADO_PY_EXE%" goto run
set "HYCADO_PY_EXE=py"
set "HYCADO_PY_SELECTOR=-3.11"
:run
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
echo HyCADO 28: Donmus YOLO-800 + YOLO-1280 cift cozunurluk fuzyonu.
echo CPU analizi. Egitim, GPU cikarimi ve pip kurulumu yok.
echo Ayni CMD, elektrik/restart kesintisinden sonra kaldigi kayitlardan devam eder.
if "%~2"=="" goto default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_fusion28.py" --project "%HYCADO_PROJECT%" --dataset "%~2"
goto check
:default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_fusion28.py" --project "%HYCADO_PROJECT%"
:check
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya bu CMD ile ayni klasordedir:
echo 28_cift_cozunurluk_fuzyon_rapor.zip
goto end
:failed
echo DURDU VEYA KESILDI. Olustuysa bu CMD yanindaki rapor ZIP dosyasini gonderin.
echo Kesinti sonrasi ayni CMD'yi calistirin. Deney klasorunu silmeyin.
:end
pause
endlocal
