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
if not exist "hycado_start\crop_training_v2.py" goto missing
if not exist "configs\10_RTDETR_Crop_Egitim_v1.json" goto missing
if not exist "experiments\rtdetr_crop_calibration_v2_seed42\result.json" goto missing
echo RT-DETR crop UZUN EGITIM: seed 42, 80 kaynak turu.
echo Her tur 5824 ornek; toplam 465920 ornek ve 58240 optimizer denemesi.
echo Olculen yalniz egitim suresi ~44.7 saat; kayit ve tanisal degerlendirme buna eklenir.
echo Her tamamlanan turda kayit alinir. Ayni CMD son tamamlanan turdan devam eder.
echo Diger GPU deneylerini kapatin; bilgisayarin otomatik uykuya gecmesini engelleyin.
".venv\Scripts\python.exe" -m hycado_start.crop_experiment
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo experiments\rtdetr_crop_v1_seed42\10_rtdetr_crop_egitim_rapor.zip
pause
exit /b 0
:missing
echo Paketin ICERIGINI mevcut D:\HyCADO_Temiz_Baslangic icine kopyalayin.
echo Mevcut ortam, veri, work ve deney klasorlerini koruyun. Basarili 09B kayitlari gerekli.
pause
exit /b 1
:failed
echo KOSU DURDU. Varsa su kucuk raporu paylasin:
echo experiments\rtdetr_crop_v1_seed42\10_rtdetr_crop_egitim_rapor.zip
echo ZIP yoksa terminal hata metnini paylasin.
echo Son tamamlanmis tur korunur. Devam icin yine bu 10 dosyasini acin.
pause
exit /b 1
