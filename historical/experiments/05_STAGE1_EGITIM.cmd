@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m hycado_start stage1
if errorlevel 1 goto failed
echo Egitim butcesi tamamlandi.
echo experiments\stage1_v1_seed42\stage1_rapor.zip dosyasini gonderin.
pause
exit /b 0
:missing
echo Python ortami bulunamadi. Dosyayi mevcut HyCADO_Temiz_Baslangic klasorunde calistirin.
pause
exit /b 1
:failed
echo EGITIM TAMAMLANMADI. experiments\stage1_v1_seed42\stage1_rapor.zip dosyasini gonderin.
echo Varsa son saglam checkpoint korunur; yoksa ayni veri ayrimiyla epoch 1 tekrar baslar.
echo ZIP olusmadiysa hata metnini kopyalayip paylasin.
pause
exit /b 1
