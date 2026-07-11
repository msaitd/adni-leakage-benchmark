@echo off
setlocal
set RL=%~dp0
echo ================================================================
echo  STEP 5 (COMBINED): CAT12 follow-up segmentation, then GPU longitudinal
echo  1) ~236 follow-up scans, CAT12 (MATLAB, long, resumable)
echo  2) when done: longitudinal manifest + 3D CNN change (GPU), automatic
echo ================================================================
echo [1/3] CAT12 follow-up segmentation...
matlab -batch "cd('%RL%'); step1d_cat12_followup"
if errorlevel 1 goto err
echo.
echo [2/3] Building longitudinal manifest...
cd /d "%RL%..\gpu_deep"
python -c "import nibabel" 2>NUL || pip install nibabel
python build_longitudinal_manifest.py || goto err
echo.
echo [3/3] Leakage-safe 3D CNN change + fusion (GPU)...
python longitudinal_change_deep.py || goto err
echo.
echo === DONE. gpu_deep\long_change_summary.csv is ready. ===
exit /b 0
:err
echo.
echo *** An error occurred - copy the last message shown above. ***
pause & exit /b 1
