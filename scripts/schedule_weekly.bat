@echo off
REM ============================================================================
REM scripts/schedule_weekly.bat
REM Skrip Eksekusi Otomatis Penjadwalan Pemantauan Isu Mingguan (Windows)
REM Dapat didaftarkan ke Windows Task Scheduler untuk dijalankan rutin tiap Senin
REM ============================================================================

set SCRIPT_DIR=%~dp0..
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo   [PEMANDUAN ISU MINGGUAN UNDIKSHA - TASK SCHEDULER]
echo   Waktu Eksekusi: %DATE% %TIME%
echo ============================================================

REM Gunakan environment python pengguna
C:\Users\Bintang\anaconda3\python.exe -m src.pipeline.weekly_run --days 7

echo.
echo Selesai. Log tercatat pada logs/weekly_pipeline.log
exit /b %ERRORLEVEL%

