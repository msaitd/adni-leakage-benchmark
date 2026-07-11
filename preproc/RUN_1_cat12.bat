@echo off
setlocal
echo ================================================================
echo  STEP 1: CAT12 segmentation (MATLAB)  -  takes a long time, resumable
echo ================================================================
matlab -batch "cd('%~dp0'); step1_cat12_segment"
if errorlevel 1 (
  echo.
  echo Could not be started from the MATLAB command line.
  echo Alternatif: MATLAB'i ac, %~dp0step1_cat12_segment.m dosyasini ac, F5'e bas.
  pause & exit /b 1
)
echo === CAT12 done. You can now run RUN_2 (in the gpu_deep folder). ===
pause
