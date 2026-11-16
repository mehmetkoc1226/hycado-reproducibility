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
echo HyCADO29: Ham crop hash/icerik denetimi ve alpha0.25 birebir yeniden uretim.
echo CPU kullanilir. Egitim, GPU cikarimi, GT/AP hesabi veya paket kurulumu yok.
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0audit29.py" --project "%HYCADO_PROJECT%"
if errorlevel 1 goto failed
echo TAMAMLANDI. Bu CMD yanindaki dosyayi gonderin:
echo 29_ham_crop_denetim_rapor.zip
goto end
:failed
echo DURDU VEYA KESILDI. Olustuysa bu CMD yanindaki rapor ZIP dosyasini gonderin.
echo Kesinti sonrasi ayni CMD ile devam edin. Deney klasorunu silmeyin.
:end
pause
endlocal
