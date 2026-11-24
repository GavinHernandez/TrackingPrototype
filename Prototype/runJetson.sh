#!/bin/bash
set -e

echo "=== PATS II Jetson Plug-and-Play Launcher ==="

# Copy project to Jetson local storage (FAT32/exFAT flash drives block executable permissions)
DEST="$HOME/PATS_Jetson"
mkdir -p "$DEST"
cp -u cameraHandoff.py yolov8n.pt jetsonRequirements.txt "$DEST/"
cd "$DEST"

# List detected cameras so you can verify CAM1_SOURCE / CAM2_SOURCE
echo "--- Connected Video Devices ---"
ls /dev/video* 2>/dev/null || echo "No USB cameras detected yet."

# Create Jetson-native venv (inherits pre-installed JetPack CUDA PyTorch & OpenCV)
if [ ! -d ".venv_jetson" ]; then
    echo "First-time setup: creating Python environment..."
    python3 -m venv --system-site-packages .venv_jetson
    source .venv_jetson/bin/activate
    pip install --upgrade pip
    pip install -r requirements_jetson.txt
else
    source .venv_jetson/bin/activate
fi

echo "Starting Dual-Camera Tracker..."
python3 cameraHandoff.py