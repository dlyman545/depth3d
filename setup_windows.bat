@echo off
REM setup_windows.bat — TripoSR setup for Windows
REM Run once: setup_windows.bat
REM Then run:  start_windows.bat

echo.
echo ╔══════════════════════════════════════════╗
echo ║   DEPTH3D · TripoSR Setup (Windows)     ║
echo ╚══════════════════════════════════════════╝
echo.

where python >nul 2>&1 || (
    echo X python not found. Install Python 3.10+ from python.org
    pause & exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

pip install --upgrade pip setuptools wheel -q

REM Check for CUDA (user must have already installed CUDA toolkit)
nvcc --version >nul 2>&1 && (
    echo CUDA detected — installing GPU PyTorch ^(CUDA 12.1^)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
) || (
    echo No CUDA — installing CPU PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
)

if not exist TripoSR (
    echo Cloning TripoSR...
    git clone https://github.com/VAST-AI-Research/TripoSR.git
)

pip install -r TripoSR\requirements.txt -q
pip install flask pillow trimesh huggingface_hub rembg -q

REM Write .pth so Python finds the tsr package
python -c "import site; open(site.getsitepackages()[0]+'/triposr.pth','w').write('%cd%\\TripoSR')"

echo.
echo ╔══════════════════════════════════════════╗
echo ║  Setup done! Run: start_windows.bat     ║
echo ╚══════════════════════════════════════════╝
pause
