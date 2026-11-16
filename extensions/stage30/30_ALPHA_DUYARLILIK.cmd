@echo off
setlocal
cd /d "%~dp0"
set "HYCADO_PROJECT=%~1"
if not defined HYCADO_PROJECT set "HYCADO_PROJECT=D:\HyCADO_Temiz_Baslangic"
set "HYCADO_PY_EXE=%HYCADO_PROJECT%\.venv\Scripts\python.exe"
set "HYCADO_PY_SELECTOR="
if exist "%HYCADO_PY_EXE%" goto run
set "HYCADO_PY_EXE=py"
set "HYCADO_PY_SELECTOR=-3.11"
:run
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
echo HyCADO30: Sabit alpha duyarliligi ve crop kutularinin GT eslesmesi.
echo CPU kullanilir. Egitim, GPU cikarimi ve pip kurulumu yok.
if "%~2"=="" goto default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_alpha30.py" --project "%HYCADO_PROJECT%" --dataset "%~2"
goto check
:default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_alpha30.py" --project "%HYCADO_PROJECT%"
:check
if errorlevel 1 goto failed
echo TAMAMLANDI. Bu CMD yanindaki dosyayi gonderin:
echo 30_alpha_duyarlilik_rapor.zip
goto end
:failed
echo DURDU VEYA KESILDI. Olustuysa bu CMD yanindaki rapor ZIP dosyasini gonderin.
echo Kesinti sonrasi ayni CMD ile devam edin. Deney klasorunu silmeyin.
:end
pause
endlocal
