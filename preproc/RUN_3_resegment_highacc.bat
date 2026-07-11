@echo off
setlocal
echo ================================================================
echo  STEP 3 (optional): re-segment low-IQR and missing scans at HIGH
echo  accuracy. Wait for RUN_1 to finish FIRST
echo  ve qc\rerun_list.csv'nin guncel oldugundan emin ol.
echo ================================================================
matlab -batch "cd('%~dp0'); step1b_resegment_highacc"
if errorlevel 1 (
  echo.
  echo Could not be started from the MATLAB command line.
  echo Alternatif: MATLAB'i ac, %~dp0step1b_resegment_highacc.m dosyasini ac, F5'e bas.
  pause & exit /b 1
)
echo === Re-segmentation done. ===
pause
