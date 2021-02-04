#!/bin/bash

echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

NAME=wendt

echo "=== Init task spooler ==="
echo "Setup task spooler socket for GPU."

export TS_SOCKET="/srv/ts_socket/GPU.socket"
chmod 777 /srv/ts_socket/GPU.socket
export TS_TMPDIR=/home/$NAME/logs
echo task spooler output directory: /home/$NAME/logs

#echo "Activate python environment and add python path variables"
#source /home/$NAME/tf2odapi/tf24/bin/activate

#echo "Add Object detection API to python path"
#cd /home/$NAME/tf2odapi/models/research
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#echo $PYTHONPATH

#echo "Go to start directory"
#echo "/home/$NAME"
#cd /home/$NAME

#echo "File End"
