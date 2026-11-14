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
echo HyCADO COCO preflight: CPU. Egitim ve GPU cikarimi yok.
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0coco_runner.py" --project "%HYCADO_PROJECT%" --dataset "%~2" --mode preflight
if errorlevel 1 goto failed
echo TAMAMLANDI. Yukarida yazan rapor ZIP dosyasini gonderin.
goto end
:failed
echo DURDU. Olustuysa bu CMD yanindaki rapor ZIP dosyasini gonderin.
:end
pause
endlocal
