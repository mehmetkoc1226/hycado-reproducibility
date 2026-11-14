@echo off
setlocal
cd /d "%~dp0"
set "HYCADO_PROJECT=%~1"
if not defined HYCADO_PROJECT set "HYCADO_PROJECT=D:\HyCADO_Temiz_Baslangic"
set "HYCADO_PY_SELECTOR="
set "HYCADO_PY_EXE=%HYCADO_PROJECT%\.venv\Scripts\python.exe"
if exist "%HYCADO_PY_EXE%" goto preflight
set "HYCADO_PY_EXE=py"
set "HYCADO_PY_SELECTOR=-3.11"
:preflight
echo HyCADO COCO v2: SAHI kontrol duzeltmesi. Egitim ve GPU cikarimi yok.
echo Once butun P2 VAL kaynak dosyalari ve kucuk metrik kontrolu dogrulanacak.
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0coco_runner.py" --project "%HYCADO_PROJECT%" --dataset "%~2" --mode preflight
if errorlevel 1 goto failed_preflight
echo ON KONTROL GECTI. Tam COCO hesabi otomatik basliyor.
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0coco_runner.py" --project "%HYCADO_PROJECT%" --dataset "%~2" --mode full
if errorlevel 1 goto failed_full
echo TAMAMLANDI. Bu CMD yanindaki 22B_coco_nokta_sonuclar_rapor.zip dosyasini gonderin.
goto end
:failed_preflight
echo ON KONTROL DURDU. Bu CMD yanindaki 22A_coco_onkontrol_rapor.zip dosyasini gonderin.
goto end
:failed_full
echo TAM HESAP DURDU. Bu CMD yanindaki 22B_coco_nokta_sonuclar_rapor.zip dosyasini gonderin.
:end
pause
endlocal
