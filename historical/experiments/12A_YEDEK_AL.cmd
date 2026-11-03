@echo off
setlocal
cd /d "%~dp0"
set "ERRORLEVEL="
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "hycado_start\project_backup.py" goto missing
echo 12A: BUTUN PROJE ICIN TARIHLI YEDEK ALINIYOR.
echo Diger HyCADO egitim ve degerlendirme pencereleri kapali olmali.
echo Hedef: proje klasorunun yanindaki HyCADO_Yedekleri klasoru.
echo Kaynak dosyalar silinmez. Her kopyanin SHA256 degeri dogrulanir.
".venv\Scripts\python.exe" -X faulthandler -u -m hycado_start.project_backup
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
".venv\Scripts\python.exe" -c "import json,sys; r=json.load(open('reports/12_yedek_sonuc.json')); sys.exit(0 if r.get('status')=='COMPLETED' and r.get('all_copied_files_sha256_verified') is True else 1)"
set "HYCADO_EXIT=%ERRORLEVEL%"
if not "%HYCADO_EXIT%"=="0" goto failed
echo YEDEK TAMAMLANDI. Simdi 12B_RESMI_VAL_DEGERLENDIR.cmd calistirin.
pause
exit /b 0
:failed
echo YEDEK TAMAMLANMADI. 12B'ye gecmeyin. Cikis kodu: %HYCADO_EXIT%
echo Hata metnini veya reports\12_yedek_sonuc.json dosyasini gonderin.
pause
exit /b 1
:missing
echo Paketi mevcut D:\HyCADO_Temiz_Baslangic klasorune kopyalayin.
pause
exit /b 2
