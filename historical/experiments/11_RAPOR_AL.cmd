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
if not exist "hycado_start\dev_evaluation.py" goto missing
if not exist "experiments\development_eval_v1_seed42\protocol.json" goto missing
".venv\Scripts\python.exe" -m hycado_start.dev_evaluation --report-only
if not "%ERRORLEVEL%"=="0" goto failed
echo Rapor: experiments\development_eval_v1_seed42\11_gelistirme_rapor.zip
pause
exit /b 0
:missing
echo Once 11_GELISTIRME_DEGERLENDIR.cmd dosyasini calistirin.
pause
exit /b 2
:failed
echo Rapor alinamadi. Terminal hata metnini paylasin.
pause
exit /b 1
