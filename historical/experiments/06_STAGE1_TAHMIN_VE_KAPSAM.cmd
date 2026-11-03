@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\stage1.py" goto missing
if not exist "experiments\stage1_v1_seed42\checkpoint_index.json" goto missing_training
echo Stage-1 epoch 80 EMA ile tahmin ve crop kapsami olculecek.
echo Yeni egitim baslatilmaz. Tamamlanan tahminler kesintide korunur.
".venv\Scripts\python.exe" -m hycado_start.proposals
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo experiments\stage1_predictions_v1_seed42\06_tahmin_kapsam_rapor.zip
pause
exit /b 0
:missing
echo Bu ek paketin ICERIGINI mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
echo .venv ve eski hycado_start dosyalari ayni proje klasorunde olmali.
echo 01-05 adimlarini tekrar calistirmayin.
pause
exit /b 1
:missing_training
echo Stage-1 checkpoint indeksi bulunamadi.
echo experiments\stage1_v1_seed42 klasorunu silmeyin veya tasimayin.
pause
exit /b 1
:failed
echo TAMAMLANAMADI. Gonderilecek dosya varsa:
echo experiments\stage1_predictions_v1_seed42\06_tahmin_kapsam_rapor.zip
echo ZIP yoksa bu penceredeki hata metnini paylasin.
echo Egitim agirliklari korunur. Ayni 06 dosyasi tamamlanmis tahminleri kullanir.
pause
exit /b 1
