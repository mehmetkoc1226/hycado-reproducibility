@echo off
setlocal
set "PROJ=D:\HyCADO_Temiz_Baslangic"
if not "%~1"=="" set "PROJ=%~1"
set "PY=%PROJ%\.venv\Scripts\python.exe"
if not exist "%PY%" goto missing
set "PYTHONUTF8=1"
echo 18A: Eski VAL ile birebir kontrol ve 12 test-dev goruntusuyle sure tahmini.
echo Egitim yok. Test-dev etiketleri acilmayacak. Diger GPU islerini kapatin.
if "%~2"=="" goto auto
"%PY%" -u "%~dp0testdev.py" preflight --project "%PROJ%" --dataset "%~2"
goto done
:auto
"%PY%" -u "%~dp0testdev.py" preflight --project "%PROJ%"
:done
set "RC=%errorlevel%"
if not "%RC%"=="0" echo On kontrol tamamlanamadi. Ekrandaki hata metnini paylasin.
pause
exit /b %RC%
:missing
echo Python bulunamadi: "%PY%"
echo Proje baska yerdeyse, ilk arguman olarak proje klasorunu verin.
pause
exit /b 1
