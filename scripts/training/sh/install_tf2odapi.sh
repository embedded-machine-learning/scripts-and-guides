#!/bin/bash

echo WARNING: Exeute this file with . and not with ./ to get the environment in the correct shell.

echo # Setup Tensorflow Object Detection API
mkdir tf2odapi
cd tf2odapi
TF2ROOT=`pwd`

echo # Create environment
virtualenv -p python3 tf24
source tf24/bin/activate

echo # install necessary software
pip install --upgrade pip
pip install tensorflow==2.4.1
echo #Test if Tensorflow works with CUDA on the machine. For TF2.4.1, you have to use CUDA 11.0
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

echo # Install protobuf
PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d protobuf
rm -f $PROTOC_ZIP

echo # Clone tensorflow repository
git clone https://github.com/tensorflow/models.git
cd models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install .

echo # Add object detection and slim to python path
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

echo # Prepare TF2 Proto Files
../../protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

echo # Test installation
# If all tests are OK or skipped, then the installation was successful
python object_detection/builders/model_builder_tf2_test.py
cd $TF2ROOT

# Install libraries for inference and visualization
echo install additional libraries
pip install numpy tdqm xmltodict pandas matplotlib pillow beautifulsoup4

echo clone scripts and guides repo for source of scripts
git clone https://github.com/embedded-machine-learning/scripts-and-guides.git

echo "Important information: If there are any library errors, you have to install the correct versions manually. TFODAPI does install the latest version of "
echo "tensorflow. However, in this script Tensorflow 2.4.1 is desired. Then, you have to uninstall the newer versions and replace with current versions."

echo # Installation complete
