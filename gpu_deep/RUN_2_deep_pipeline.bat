@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"

echo ================================================================
echo  STEP 2: manifest, smoke test, full training, leakage-safe fusion
echo ================================================================

where python >nul 2>nul
if errorlevel 1 (
  echo *** ERROR: Python was not found on PATH. ***
  goto err
)

echo [1/4] Building the baseline manifest...
python -u build_deep_manifest.py
if errorlevel 1 goto err

echo [2/4] Duman testi yapiliyor...
python -u train_cnn_cv.py --smoke
if errorlevel 1 goto err

echo [3/4] Full training starting...
echo       Tamamlanan foldlar dogrulanir ve tekrar egitilmez.
python -u train_cnn_cv.py
if errorlevel 1 goto err

echo [4/4] Outer-fold uyumlu, sizintisiz fuzyon ve rapor...
python -u fuse_and_report.py
if errorlevel 1 goto err

echo.
echo ================================================================
echo  DONE
echo  Modeller : fixed_results\
echo  Ozet     : fixed_deep_final_summary.csv
echo  OOF      : fixed_deep_final_oof.csv
echo  Sekil    : fixed_deep_final_comparison.png
echo ================================================================
pause
exit /b 0

:err
set "EXITCODE=%ERRORLEVEL%"
if "%EXITCODE%"=="0" set "EXITCODE=1"
echo.
echo *** An error occurred. Keep the last error message shown above. ***
echo *** Cikis kodu: %EXITCODE% ***
pause
exit /b %EXITCODE%
