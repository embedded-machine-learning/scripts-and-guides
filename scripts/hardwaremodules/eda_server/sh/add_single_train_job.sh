#!/bin/sh

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
MODELNAME=tf2oda_efficientdetd0_512x320_pedestrian_LR02
PYTHONENV=tf24
BASEPATH=.
SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training
CURRENTFOLDER=`pwd`

echo "Setup task spooler socket."
. ~/init_eda_ts.sh

MODELNAME1=$MODELNAME


echo "Add task spooler jobs to the task spooler"
ts -L AW_$MODELNAME1 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME1.sh

