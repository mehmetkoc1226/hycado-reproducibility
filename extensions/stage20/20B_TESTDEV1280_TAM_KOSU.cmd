@echo off
setlocal
set "PROJ=D:\HyCADO_Temiz_Baslangic"
if not "%~1"=="" set "PROJ=%~1"
set "PY=%PROJ%\.venv\Scripts\python.exe"
if not exist "%PY%" goto missing
set "PYTHONUTF8=1"
echo 20: Test-dev 1280 varyantlari. Yeni egitim, yeni veri indirme yok.
echo Onceki 18 ve 19 klasorleri gerekli. Diger GPU islerini kapatin.
if "%~2"=="" goto auto
"%PY%" -u "%~dp0test1280.py" full --project "%PROJ%" --dataset "%~2"
goto done
:auto
"%PY%" -u "%~dp0test1280.py" full --project "%PROJ%"
:done
set "RC=%errorlevel%"
if not "%RC%"=="0" echo Islem durdu. Ekrandaki hata metnini veya error.json dosyasini paylasin.
pause
exit /b %RC%
:missing
echo Python bulunamadi: "%PY%"
pause
exit /b 1
