#!/bin/bash

# put in home

echo "WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./"
echo "WARNING: Set variable ENVROOT to your root for environment and TF2ODA models."

#echo "Setup task spooler socket for GPU."

#export TS_SOCKET="/srv/ts_socket/GPU.socket"
#chmod 777 /srv/ts_socket/GPU.socket
#export TS_TMPDIR=/home/$NAME/logs
#echo task spooler output directory: /home/$NAME/logs

#echo "Setup Task Spooler"
#. /srv/cdl-eml/tf2odapi/init_eda_ts.sh

# Load VSC modules
module purge #clear
module load anaconda3/5.3.0 cuda/11.0.2 nvidia/1.0 #Load models that we need

echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'Using the nodes: '$SLURM_JOB_NODELIST


echo "Activate TF2 Object Detection API Python Environment"

TF2ROOT=`pwd`
ENVROOT=/home/lv71513/awendt/eml

#source $ENVROOT/tf24/bin/activate
source /opt/sw/x86_64/glibc-2.17/ivybridge-ep/anaconda3/5.3.0/etc/profile.d/conda.sh
#source /home/${USER}/.bashrc
#conda init bash
conda activate tf24

cd $ENVROOT/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo New python path $PYTHONPATH

cd $TF2ROOT

echo "Activation complete"
