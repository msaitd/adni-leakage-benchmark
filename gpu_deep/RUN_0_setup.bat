@echo off
echo ================================================================
echo  STEP 0: Setup (PyTorch CUDA + MONAI)  [ONCE]
echo ================================================================
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install monai nibabel scikit-learn pandas matplotlib openpyxl
echo.
echo === GPU kontrolu ===
python -c "import torch;print('CUDA available:',torch.cuda.is_available());print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND - update the driver')"
echo.
pause
