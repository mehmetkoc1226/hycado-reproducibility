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
echo HyCADO 23: CPU guven araliklari ve recall-FP. Egitim veya GPU cikarimi yok.
echo Mevcut 22B cache ve 20 eslesmeleri okunacak. Tamamlanan tekrarlar korunur.
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0run_analysis.py" --project "%HYCADO_PROJECT%"
if errorlevel 1 goto failed
echo TAMAMLANDI. Bu CMD yanindaki 23_coco_guven_recall_rapor.zip dosyasini gonderin.
goto end
:failed
echo DURDU VEYA KESILDI. Olustuysa bu CMD yanindaki 23_coco_guven_recall_rapor.zip dosyasini gonderin.
echo Kesinti sonrasi ayni CMD dosyasi ile devam edilebilir.
:end
pause
endlocal
