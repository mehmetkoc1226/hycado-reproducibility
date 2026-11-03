@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\dev_diagnostics.py" goto missing
if not exist "experiments\development_eval_v1_seed42\result.json" goto missing
".venv\Scripts\python.exe" -m hycado_start.dev_diagnostics --report-only
if not "%ERRORLEVEL%"=="0" goto failed
echo Rapor: experiments\development_diagnostics_v1_seed42\11B_tani_rapor.zip
pause
exit /b 0
:missing
echo Once paketi mevcut proje klasorune kopyalayip 11B_CPU_TANI.cmd dosyasini calistirin.
pause
exit /b 2
:failed
echo Rapor alinamadi. Terminal hata metnini paylasin.
pause
exit /b 1
