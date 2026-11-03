@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m hycado_start stability
if errorlevel 1 goto failed
echo Testler bitti. reports klasorundeki en yeni stability_..._rapor.zip dosyasini gonderin.
pause
exit /b 0
:missing
echo Proje ortami bulunamadi. Bu dosyayi mevcut HyCADO_Temiz_Baslangic klasorunde calistirin.
pause
exit /b 1
:failed
echo Islem hata verdi. reports klasorundeki en yeni stability_..._rapor.zip dosyasini gonderin.
echo ZIP olusmadiysa bu penceredeki hata metnini kopyalayip paylasin.
pause
exit /b 1
