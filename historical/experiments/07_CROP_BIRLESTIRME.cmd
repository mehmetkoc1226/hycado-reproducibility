@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\proposals.py" goto missing
if not exist "experiments\stage1_predictions_v1_seed42\prediction_index.json" goto missing06
echo CPU ile crop birlestirme ve buyutme olcumu basliyor.
echo Mevcut 06 tahminleri kullanilir. Model yuklenmez, egitim yapilmaz.
".venv\Scripts\python.exe" -m hycado_start.crop_merge
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo experiments\crop_merge_v1_seed42\07_crop_birlestirme_rapor.zip
pause
exit /b 0
:missing
echo Bu ek paketin ICERIGINI mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo .venv ve onceki 06 kodlari ayni proje klasorunde olmali.
pause
exit /b 1
:missing06
echo 06 tahmin indeksi bulunamadi.
echo experiments\stage1_predictions_v1_seed42 klasorunun yerini kontrol edin.
echo Mevcut egitim ve tahmin klasorlerini silmeyin.
pause
exit /b 1
:failed
echo TAMAMLANAMADI. Varsa su raporu gonderin:
echo experiments\crop_merge_v1_seed42\07_crop_birlestirme_rapor.zip
echo ZIP yoksa hata metnini paylasin.
echo Kaynak egitim/tahmin dosyalari korunur. Ayni 07 dosyasindan devam edilebilir.
pause
exit /b 1
