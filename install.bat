@echo off
REM Tripedal Robot RL Project Installation Script for Windows

echo ==========================================
echo Tripedal Robot RL Project Installation
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8+.
    pause
    exit /b 1
)

echo Python found

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip is not installed or not in PATH. Please install pip.
    pause
    exit /b 1
)

echo pip found

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Install the project in development mode
echo Installing project in development mode...
pip install -e .

echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo IMPORTANT: This project requires NVIDIA Isaac Sim to be installed.
echo Please follow the official Isaac Sim installation guide:
echo https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html
echo.
echo After installing Isaac Sim, you can run your RL training scripts.
echo.
echo Example usage:
echo   python scripts/reinforcement_learning/rsl_rl/train.py --task=triiple_legs_robot
echo.
pause 