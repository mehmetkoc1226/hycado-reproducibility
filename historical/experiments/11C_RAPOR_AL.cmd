@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONHASHSEED=42
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\crop_capture.py" goto missing
if not exist "hycado_start\dev_diagnostics.py" goto missing
if not exist "experiments\development_eval_v1_seed42\result.json" goto missing
".venv\Scripts\python.exe" -m hycado_start.crop_capture --report-only
if not "%ERRORLEVEL%"=="0" goto failed
echo Rapor: experiments\raw_crop_capture_v1_seed42\11C_ham_crop_rapor.zip
pause
exit /b 0
:missing
echo Once 11C paketini mevcut proje klasorune kopyalayin.
pause
exit /b 2
:failed
echo Rapor alinamadi. Terminal hata metnini paylasin.
pause
exit /b 1
