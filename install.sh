#!/bin/bash

# Tripedal Robot RL Project Installation Script

echo "=========================================="
echo "Tripedal Robot RL Project Installation"
echo "=========================================="

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    echo "✓ Python $python_version found"
else
    echo "✗ Python 3.8+ is required. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 found"
else
    echo "✗ pip3 is required. Please install pip3."
    exit 1
fi

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install the project in development mode
echo "Installing project in development mode..."
pip3 install -e .

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: This project requires NVIDIA Isaac Sim to be installed."
echo "Please follow the official Isaac Sim installation guide:"
echo "https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html"
echo ""
echo "After installing Isaac Sim, you can run your RL training scripts."
echo ""
echo "Example usage:"
echo "  python scripts/reinforcement_learning/rsl_rl/train.py --task=triiple_legs_robot"
echo "" 