#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a package is installed
package_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -c "ok installed"
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python is not installed. Installing the latest version of Python..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip
else
    echo "Python is already installed."
    sudo apt update
    sudo apt upgrade -y python3 python3-venv python3-pip
fi

# Check if python3-venv is installed
if [ "$(package_installed python3.10-venv)" -eq 0 ]; then
    echo "python3-venv is not installed. Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3.10-venv
else
    echo "python3-venv is already installed. Upgrading to the latest version..."
    sudo apt update
    sudo apt upgrade -y python3-venv
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment does not exist. Creating a new virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists. Upgrading pip and setuptools..."
    source .venv/bin/activate
    pip install --upgrade pip setuptools
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
mkdir -p src
mkdir -p src/data_preprocess
mkdir -p src/data_preprocess/nist
mkdir -p src/data_preprocess/nist/data_root
mkdir -p graph
mkdir -p src/data_preprocess/dataset
mkdir -p src/data_preprocess/dataset/off
mkdir -p src/data_preprocess/dataset/on
mkdir -p src/data_preprocess/dataset/test
mkdir -p src/data_preprocess/dataset/train

declare -A datasets

datasets[src/data_preprocess/nist/data_root/HVAC-minute-2014.csv]="https://s3.amazonaws.com/nist-netzero/2014-data-files/HVAC-minute.csv"
datasets[src/data_preprocess/nist/data_root/HVAC-minute-2015.csv]="https://s3.amazonaws.com/nist-netzero/2015-data-files/HVAC-minute.csv"
datasets[src/data_preprocess/nist/data_root/IndEnv-minute-2014.csv]="https://s3.amazonaws.com/nist-netzero/2014-data-files/IndEnv-minute.csv"
datasets[src/data_preprocess/nist/data_root/IndEnv-minute-2015.csv]="https://s3.amazonaws.com/nist-netzero/2015-data-files/IndEnv-minute.csv"
datasets[src/data_preprocess/nist/data_root/OutEnv-minute-2014.csv]="https://s3.amazonaws.com/nist-netzero/2014-data-files/OutEnv-minute.csv"
datasets[src/data_preprocess/nist/data_root/OutEnv-minute-2015.csv]="https://s3.amazonaws.com/nist-netzero/2015-data-files/OutEnv-minute.csv"

for key in "${!datasets[@]}" 
do 
    if [ ! -f "$key" ]; then
        echo "Downloading $key..."
        wget -O $key "${datasets[$key]}"
    else
        echo "$key already exists."
    fi
done

echo "Setup complete."