@echo off
setlocal
set "PROJ=D:\HyCADO_Temiz_Baslangic"
if not "%~1"=="" set "PROJ=%~1"
set "PY=%PROJ%\.venv\Scripts\python.exe"
if not exist "%PY%" goto missing
set "PYTHONUTF8=1"
echo HyCADO1280 - 647 TRAIN ici gelistirme. Yeni egitim ve yeni veri indirme yok.
echo Diger GPU islerini kapatin. Kesinti olursa ayni CMD'yi yeniden acin.
"%PY%" -u "%~dp0run1280.py" full --project "%PROJ%"
set "RC=%errorlevel%"
if not "%RC%"=="0" echo Islem durdu. Ekrandaki hata metnini veya error.json dosyasini paylasin.
pause
exit /b %RC%
:missing
echo Python bulunamadi: "%PY%"
echo Proje baska yerdeyse ilk argumanla proje klasorunu verin.
pause
exit /b 1
