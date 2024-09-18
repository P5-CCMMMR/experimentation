#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python is not installed. Installing the latest version of Python..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip
else
    echo "Python is already installed."
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment does not exist. Creating a new virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source .venv/bin/activate

# Install libraries from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing libraries from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt file not found."
fi

# Download datasets from NIST
echo "Downloading datasets from NIST..."
mkdir -p dataset
wget -O dataset/HVAC-minute.csv https://s3.amazonaws.com/nist-netzero/2014-data-files/HVAC-minute.csv
wget -O dataset/IndEnv-minute.csv https://s3.amazonaws.com/nist-netzero/2014-data-files/IndEnv-minute.csv
wget -O dataset/OutEnv-minute.csv https://s3.amazonaws.com/nist-netzero/2014-data-files/OutEnv-minute.csv

echo "Setup complete."