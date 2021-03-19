#!/bin/sh

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
PYTHONENV=tf24
BASEPATH=.
SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training
CURRENTFOLDER=`pwd`

echo "Setup task spooler socket."
. ~/init_eda_ts.sh

MODELNAME1=tf2oda_ssdmobilenetv2_320x256_pedestrian
MODELNAME2=tf2oda_ssdmobilenetv2_256x256_pedestrian
#MODELNAME3=tf2oda_efficientdetd1_768x640_pedestrian
#MODELNAME4=tf2oda_efficientdetd2_512x384_pedestrian
#MODELNAME5=tf2oda_efficientdetd2_640x480_pedestrian
#MODELNAME6=tf2oda_ssdmobilenetv2_768x576_pedestrian_D100
#MODELNAME7=
#MODELNAME8=

echo "Add task spooler jobs to the task spooler"
ts -L AW_$MODELNAME1 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME1.sh
ts -L AW_$MODELNAME2 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME2.sh
#ts -L AW_$MODELNAME3 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME3.sh
#ts -L AW_$MODELNAME4 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME4.sh
#ts -L AW_$MODELNAME5 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME5.sh
#ts -L AW_$MODELNAME1 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME6.sh
#ts -L AW_$MODELNAME1 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME7.sh
#ts -L AW_$MODELNAME1 $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME8.sh

