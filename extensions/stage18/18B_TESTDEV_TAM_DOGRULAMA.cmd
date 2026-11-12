@echo off
setlocal
set "PROJ=D:\HyCADO_Temiz_Baslangic"
if not "%~1"=="" set "PROJ=%~1"
set "PY=%PROJ%\.venv\Scripts\python.exe"
if not exist "%PY%" goto missing
if not exist "%PROJ%\experiments\testdev_frozen_v1_seed42\18_testdev_onkontrol.json" goto no_pilot
set "PYTHONUTF8=1"
echo 18B: VisDrone test-dev, 1610 goruntu. Egitim yok.
echo Diger GPU islerini kapatin. Kesinti olursa bu CMD'yi tekrar calistirin.
set "HELDOUT="
set /p "HELDOUT=Bu test-dev bolumu daha once model, esik veya ayar secimi icin KULLANILMADIYSA EVET yazin: "
if /i not "%HELDOUT%"=="EVET" goto cancelled
if "%~2"=="" goto auto
"%PY%" -u "%~dp0testdev.py" full --project "%PROJ%" --dataset "%~2" --heldout EVET
goto done
:auto
"%PY%" -u "%~dp0testdev.py" full --project "%PROJ%" --heldout EVET
:done
set "RC=%errorlevel%"
if not "%RC%"=="0" echo Kosu tamamlanamadi. Ekrandaki hata metnini veya full_error.json dosyasini paylasin.
pause
exit /b %RC%
:missing
echo Python bulunamadi: "%PY%"
pause
exit /b 1
:no_pilot
echo Once 18A_TESTDEV_ON_KONTROL.cmd dosyasini calistirin.
pause
exit /b 1
:cancelled
echo Tam kosu baslatilmadi.
pause
exit /b 1
