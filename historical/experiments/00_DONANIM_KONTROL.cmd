@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "reports" mkdir "reports"
echo HyCADO donanim kontrolu basliyor. Python veya paket kurmaz.
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $r=New-Object System.Collections.Generic.List[string]; $r.Add('HyCADO donanim kontrolu - duzeltme 1'); $r.Add('Tarih: '+(Get-Date -Format o)); $os=Get-CimInstance Win32_OperatingSystem; $r.Add('Isletim sistemi: '+$os.Caption+' '+$os.Version); $pc=Get-CimInstance Win32_ComputerSystem; $r.Add('RAM GiB: '+[math]::Round($pc.TotalPhysicalMemory/1GB,2)); $r.Add('Bos disk alani:'); Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | ForEach-Object { $r.Add($_.DeviceID+' '+[math]::Round($_.FreeSpace/1GB,2)+' GiB bos') }; [IO.File]::WriteAllLines((Join-Path (Get-Location).Path 'reports\00_donanim_raporu.txt'),$r,(New-Object Text.UTF8Encoding($true)))"
if errorlevel 1 goto failed
if not exist "reports\00_donanim_raporu.txt" goto failed
>>"reports\00_donanim_raporu.txt" echo.
>>"reports\00_donanim_raporu.txt" echo GPU, toplam bellek, bos bellek, surucu:
where nvidia-smi >nul 2>&1
if errorlevel 1 goto no_gpu_tool
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader >>"reports\00_donanim_raporu.txt" 2>&1
if errorlevel 1 >>"reports\00_donanim_raporu.txt" echo GPU sorgusu hata verdi; ustteki mesaji paylasin.
goto python_check
:no_gpu_tool
>>"reports\00_donanim_raporu.txt" echo nvidia-smi bulunamadi.
:python_check
>>"reports\00_donanim_raporu.txt" echo.
>>"reports\00_donanim_raporu.txt" echo Python launcher:
where py >nul 2>&1
if errorlevel 1 goto no_python_launcher
py -0p >>"reports\00_donanim_raporu.txt" 2>&1
if errorlevel 1 >>"reports\00_donanim_raporu.txt" echo Python listeleme tamamlanamadi; ustteki mesaj kaydedildi.
goto success
:no_python_launcher
>>"reports\00_donanim_raporu.txt" echo py launcher bulunamadi; Python kurulu olmadigi anlamina tek basina gelmez.
:success
echo.
type "reports\00_donanim_raporu.txt"
echo.
echo Rapor: reports\00_donanim_raporu.txt
echo Simdi yalnizca bu dosyayi sohbete yukleyin.
echo.
pause
exit /b 0
:failed
echo Donanim kontrolu tamamlanamadi. Bu pencerenin hata metnini paylasin.
pause
exit /b 1
