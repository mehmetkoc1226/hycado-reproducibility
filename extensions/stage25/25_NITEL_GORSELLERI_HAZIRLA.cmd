@echo off
setlocal
cd /d "%~dp0"
set "HYCADO_PROJECT=%~1"
if not defined HYCADO_PROJECT set "HYCADO_PROJECT=D:\HyCADO_Temiz_Baslangic"
set "HYCADO_PY_EXE=%HYCADO_PROJECT%\.venv\Scripts\python.exe"
set "HYCADO_PY_SELECTOR="
if exist "%HYCADO_PY_EXE%" goto run
set "HYCADO_PY_EXE=py"
set "HYCADO_PY_SELECTOR=-3.11"
:run
echo HyCADO 25 DUZELTME 1: Windows SQLite baglantilari kapatiliyor.
echo Egitim, GPU cikarimi ve paket kurulumu yok. Eski dosyalar salt okunur.
if "%~2"=="" goto default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0qualitative25.py" --project "%HYCADO_PROJECT%" --dataset "%~2"
goto check
:default_data
"%HYCADO_PY_EXE%" %HYCADO_PY_SELECTOR% "%~dp0qualitative25.py" --project "%HYCADO_PROJECT%"
:check
if errorlevel 1 goto failed
echo TAMAMLANDI. Gonderilecek dosya:
echo %HYCADO_PROJECT%\experiments\qualitative1280_v1_seed42\25_nitel_gorsel_rapor.zip
goto end
:failed
echo DURDU. Bu CMD yanindaki 25_nitel_gorsel_hata.txt dosyasini gonderin.
:end
pause
endlocal
