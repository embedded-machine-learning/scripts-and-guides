#!/bin/bash

echo "WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./"
echo "WARNING: Set variable ENVROOT to your root for environment and TF2ODA models."

#echo "Setup task spooler socket for GPU."

#export TS_SOCKET="/srv/ts_socket/GPU.socket"
#chmod 777 /srv/ts_socket/GPU.socket
#export TS_TMPDIR=/home/$NAME/logs
#echo task spooler output directory: /home/$NAME/logs

echo "Activate TF2 Object Detection API Python Environment"

TF2ROOT=`pwd`
ENVROOT=~/tf2odapi

source $ENVROOT/tf24/bin/activate
cd $ENVROOT/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo New python path $PYTHONPATH

cd $TF2ROOT

echo "Activation complete"
