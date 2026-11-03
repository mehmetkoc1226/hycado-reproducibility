@echo off
setlocal
cd /d "%~dp0"
echo Internet gerekir. Bu islem yalnizca bu klasordeki .venv ortamini kurar.
py -3.11 -c "import sys; print(sys.version)" >nul 2>&1
if not errorlevel 1 goto py311
py -3.12 -c "import sys; print(sys.version)" >nul 2>&1
if not errorlevel 1 goto py312
echo Python 3.11 veya 3.12 launcher ile bulunamadi.
echo Ilk donanim raporunu gonderin; kurulum yolunu ona gore belirleyelim.
pause
exit /b 1
:py311
py -3.11 scripts\setup_env.py
goto done
:py312
py -3.12 scripts\setup_env.py
:done
if errorlevel 1 goto failed
echo Kurulum tamamlandi. reports\01_python_cuda_kontrol.json olustu.
pause
exit /b 0
:failed
echo Kurulum tamamlanamadi. Hata metnini paylasin.
pause
exit /b 1
