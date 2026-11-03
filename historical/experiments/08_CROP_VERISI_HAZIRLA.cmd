@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\crop_merge.py" goto missing
if not exist "configs\HyCADO_Crop_Politikasi_v1.json" goto missingjson
if not exist "experiments\crop_merge_v1_seed42\merge_summary.json" goto missing07
echo Secilmis crop politikasi ile gercek PNG ve anotasyon verisi uretilecek.
echo Model yuklenmez. Egitim baslatilmaz. Veriler D uzerinde mevcut proje icine yazilir.
".venv\Scripts\python.exe" -m hycado_start.crop_data
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo experiments\crop_dataset_v1_seed42\08_crop_veri_rapor.zip
pause
exit /b 0
:missing
echo Ek paketin ICERIGINI mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo Mevcut .venv, data, experiments ve onceki kod klasorlerini koruyun.
pause
exit /b 1
:missingjson
echo Paketin configs klasoru kopyalanmamis.
echo JSON'u elle aramaniza gerek yok. ZIP icindeki configs klasorunu projeye kopyalayin.
pause
exit /b 1
:missing07
echo Tamamlanmis 07 raporu bulunamadi.
echo experiments\crop_merge_v1_seed42 klasorunun mevcut oldugunu kontrol edin.
pause
exit /b 1
:failed
echo TAMAMLANAMADI. Varsa su kucuk raporu gonderin:
echo experiments\crop_dataset_v1_seed42\08_crop_veri_rapor.zip
echo ZIP yoksa hata metnini paylasin.
echo Kaynaklar ve tamamlanmis crop klasorleri korunur. Ayni 08 dosyasindan devam edilebilir.
pause
exit /b 1
