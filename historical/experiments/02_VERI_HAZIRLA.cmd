@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m hycado_start prepare
if errorlevel 1 goto failed
echo Veri hazir. Once reports\02_veri_kalite_raporu.json dosyasini paylasin.
pause
exit /b 0
:missing
echo Once 01_KURULUM.cmd tamamlanmali.
pause
exit /b 1
:failed
echo Veri hazirligi tamamlanamadi. reports\02_veri_hazirlama.log dosyasini paylasin.
pause
exit /b 1
