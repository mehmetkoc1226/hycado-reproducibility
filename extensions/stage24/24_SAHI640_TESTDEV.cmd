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
echo HyCADO 24: Gercek SAHI640 test-dev cikarimi. Egitim yok.
echo Once kaynaklar ve 5 eski VAL ciktisi dogrulanacak. Diger GPU islerini kapatin.
if "%~2"=="" goto default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_sahi24.py" --project "%HYCADO_PROJECT%" --dataset "%~2"
goto check
:default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_sahi24.py" --project "%HYCADO_PROJECT%"
:check
if errorlevel 1 goto failed
echo TAMAMLANDI. Bu CMD yanindaki 24_sahi640_testdev_rapor.zip dosyasini gonderin.
goto end
:failed
echo DURDU VEYA KESILDI. Olustuysa bu CMD yanindaki 24_sahi640_testdev_rapor.zip dosyasini gonderin.
echo Kesinti sonrasi ayni CMD ile devam edilebilir. Eski kayitlari silmeyin.
:end
pause
endlocal
