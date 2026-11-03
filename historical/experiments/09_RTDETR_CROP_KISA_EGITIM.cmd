@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONHASHSEED=42
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\crop_data.py" goto missing
if not exist "experiments\crop_dataset_v1_seed42\source_index.json" goto missing08
echo RT-DETR gercek crop verisiyle 3 kisa egitim blogu calistiracak.
echo Toplam 768 egitim ornegi; 960x960, fiziksel batch 1, etkin batch 8.
echo GPU bos olsun. Diger egitim ve tahmin pencerelerini kapatin.
".venv\Scripts\python.exe" -m hycado_start.crop_training
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo experiments\rtdetr_crop_calibration_v1_seed42\09_rtdetr_crop_rapor.zip
pause
exit /b 0
:missing
echo Ek paketin ICERIGINI mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Mevcut .venv, data, work, experiments ve onceki kod klasorlerini koruyun.
pause
exit /b 1
:missing08
echo Tamamlanmis 08 crop verisi bulunamadi.
echo experiments\crop_dataset_v1_seed42 klasorunu koruyun. 08'i yeniden uretmeyin.
pause
exit /b 1
:failed
echo TAMAMLANAMADI. Varsa su kucuk raporu gonderin:
echo experiments\rtdetr_crop_calibration_v1_seed42\09_rtdetr_crop_rapor.zip
echo ZIP yoksa terminal hata metnini paylasin.
echo Ayni 09 dosyasini yeniden acmak son tamamlanan bloktan devam eder.
pause
exit /b 1
