#!/bin/bash
# Create and activate virtual environment for XKCD ice cream plotting

if [ ! -d "xkcd_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv xkcd_env
fi

echo "Activating virtual environment..."
source xkcd_env/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Environment ready! Run: python ice_cream_plot.py"
