@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo  STEP 5b: Image-based LONGITUDINAL change (3D CNN, GPU)
echo ================================================================
python -c "import nibabel" 2>NUL || pip install nibabel
echo [1/2] Longitudinal manifest (matches baseline + follow-up)...
python build_longitudinal_manifest.py || goto err
echo [2/2] CNN training + leakage-safe fusion (change vs baseline vs clinical)...
python longitudinal_change_deep.py || goto err
echo === DONE. long_change_summary.csv is ready. ===
pause & exit /b 0
:err
echo *** ERROR - copy and report the last message. ***
pause & exit /b 1
