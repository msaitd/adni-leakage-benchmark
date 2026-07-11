@echo off
setlocal
echo ================================================================
echo  STEP 5a: segment the FOLLOW-UP scans with CAT12 (~236 scans)
echo  based on process_list_followup.csv. Baselines are already segmented.
echo  Takes a long time (~10-20 min per scan); resumable.
echo ================================================================
matlab -batch "cd('%~dp0'); step1d_cat12_followup"
if errorlevel 1 ( echo MATLAB baslatilamadi. GUI'den step1d_cat12_followup.m ac, F5. & pause & exit /b 1 )
echo === Follow-up segmentation done. Now run gpu_deep\RUN_5b_longitudinal.bat ===
pause
