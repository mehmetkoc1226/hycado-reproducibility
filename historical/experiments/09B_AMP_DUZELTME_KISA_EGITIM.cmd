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
if not exist "hycado_start\crop_criterion.py" goto missing
if not exist "configs\09_AMP_Duzeltme_Referansi_v2.json" goto missing
if not exist "experiments\crop_dataset_v1_seed42\source_index.json" goto missing
echo 09 v2: AMP baslangic olcegi 16, ayrintili adim kaydi.
echo Ayni 768 ornek ile resmi COCO agirligindan yeni kisa kosu.
echo Eski 09 kosusu ve 08 crop verisi korunur.
echo Gereksiz programlari ve diger GPU egitimlerini kapatin.
".venv\Scripts\python.exe" -m hycado_start.crop_training_v2
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo experiments\rtdetr_crop_calibration_v2_seed42\09_rtdetr_crop_v2_rapor.zip
pause
exit /b 0
:missing
echo Paketin ICERIGINI mevcut D:\HyCADO_Temiz_Baslangic icine kopyalayin.
echo Onceki kod, ortam, agirlik ve deney klasorlerini koruyun. 01-08'i yeniden calistirmayin.
pause
exit /b 1
:failed
echo TAMAMLANAMADI. Varsa su raporu paylasin:
echo experiments\rtdetr_crop_calibration_v2_seed42\09_rtdetr_crop_v2_rapor.zip
echo ZIP yoksa hata metnini paylasin.
echo Yeniden denemek icin yine bu 09B dosyasini acin.
pause
exit /b 1
