@echo off
setlocal
set "ROOT=D:\HyCADO_Temiz_Baslangic"
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
 echo Proje Python bulunamadi: "%PY%"
 pause
 exit /b 1
)
echo CPU kontrolu: kayitli YOLO-1280 ve RT-DETR crop tahminleri.
"%PY%" -u "%~dp0fusion1280.py" --project "%ROOT%"
set "RC=%ERRORLEVEL%"
echo Rapor: "%ROOT%\experiments\p4_yolo1280_cpu_fusion_v1_seed42\14_yolo1280_cpu_fuzyon_rapor.zip"
pause
exit /b %RC%
