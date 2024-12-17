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
    apt update
    apt install -y python3 python3-venv python3-pip unzip
else
    echo "Python is already installed."
    apt update
    apt upgrade -y python3 python3-venv python3-pip
fi

# Check if python3-venv is installed
if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "python3-venv is not installed. Installing python3-venv..."
    apt update
    apt install -y python3-venv
else
    echo "python3-venv is already installed. Upgrading to the latest version..."
    apt update
    apt upgrade -y python3-venv
fi

# Check if wget is installed
if ! command_exists wget; then
    echo "wget is not installed. Installing wget..."
    apt update
    apt install -y wget
else
    echo "wget is already installed."
fi

# Check if unzip is installed
if ! command_exists unzip; then
    echo "unzip is not installed. Installing unzip..."
    apt update
    apt install -y unzip
else
    echo "unzip is already installed."
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment does not exist. Creating a new virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Check if gdown is installed within the virtual environment
if ! command_exists gdown; then
    echo "gdown is not installed. Installing gdown..."
    .venv/bin/pip install gdown
else
    echo "gdown is already installed."
fi

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

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

declare -A datasets

datasets[src/data_preprocess/nist/data_root/HVAC-minute-2014.csv]="https://s3.amazonaws.com/nist-netzero/2014-data-files/HVAC-minute.csv"
datasets[src/data_preprocess/nist/data_root/HVAC-minute-2015.csv]="https://s3.amazonaws.com/nist-netzero/2015-data-files/HVAC-minute.csv"
datasets[src/data_preprocess/nist/data_root/DHW-minute-2014.csv]="https://s3.amazonaws.com/nist-netzero/2014-data-files/DHW-minute.csv"
datasets[src/data_preprocess/nist/data_root/DHW-minute-2015.csv]="https://s3.amazonaws.com/nist-netzero/2015-data-files/DHW-minute.csv"
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

# Download datasets from UKDATA

echo "Downloading datasets from UKDATA..."
mkdir -p src/data_preprocess/ukdata
gdown 1Fv2xtlxWNabYTLpPyBVL__GR6mQ2T0MO -O src/data_preprocess/ukdata/
unzip src/data_preprocess/ukdata/UKDATA_CLEANED.zip -d src/data_preprocess/ukdata/
mv src/data_preprocess/ukdata/UKDATA_CLEANED src/data_preprocess/ukdata/data_root
rm src/data_preprocess/ukdata/UKDATA_CLEANED.zip

mkdir -p src/experiments/iter1/model_saves

echo "Setup complete."