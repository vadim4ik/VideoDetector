# VideoDetector

## Install Python 3.11
brew install python@3.11
brew install ffmpeg

## Install and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

## Install needed libraries
pip install --upgrade pip
pip install torch opencv-python ultralytics
pip install "numpy<2"
pip install deepface
pip install tf-keras

## Run script
python detect_objects.py