@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m hycado_start probe
if errorlevel 1 goto failed
echo Testler bitti. reports klasorundeki en yeni pilot_..._rapor.zip dosyasini gonderin.
pause
exit /b 0
:missing
echo Once 01_KURULUM.cmd tamamlanmali.
pause
exit /b 1
:failed
echo Test yoneticisi tamamlanamadi. Hata metnini paylasin.
pause
exit /b 1
