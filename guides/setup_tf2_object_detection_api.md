# Tensorflow Object Detection API Setup and Run

## Install Script

For Linux, use the following Script to setup TF2 OD API. The script is located in ../scripts/training/install_tf2odapi.sh

```
#!/bin/bash

echo # Setup Tensorflow Object Detection API
cd ~
mkdir tf2odapi
cd tf2odapi
TF2ROOT=`pwd`

echo # Create environment
virtualenv -p python3 tf24
source tf24/bin/activate

echo # install necessary software
pip install --upgrade pip
pip install tensorflow
echo #Test if Tensorflow works with CUDA on the machine. For TF2.4.1, you have to use CUDA 11.0
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

echo # Install protobuf
PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d Protobuf
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
python object_detection/builders/model_builder_tf2_test.py
cd $TF2ROOT

echo # Installation complete
```

## Start Script on User Login
For Linux, a start script has to be run every time to load the environment and the variables. The script is located here: ..scripts/training/init_eda_env.sh

```
#!/bin/bash

echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

echo "Activate TF2 Object Detection API Python Environment"

TF2ROOT=`pwd`
source ~/tf2odapi/tf24/bin/activate
cd ~/object_detection/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "New python path $PYTHONPATH"
cd $TF2ROOT

echo "Activation complete"
```