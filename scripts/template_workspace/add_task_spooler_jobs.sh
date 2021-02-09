#!/bin/sh

#Constants
NAME=wendt

echo "Setup task spooler socket."
. ~/init_eda_ts.sh

DATASETFOLDER=eda01_validation_tf2_star_wars

MODELNAME1=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
#MODELNAME2=ssd_mobilenet_v2_R400x400_D100_coco17_starwars
#MODELNAME3=
#MODELNAME4=
#MODELNAME5=
#MODELNAME6=
#MODELNAME7=
#MODELNAME8=

echo "Add task spooler jobs to the task spooler"
ts -L AW_$MODELNAME1 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME1.sh
#ts -L AW_$MODELNAME2 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME2.sh
#ts -L AW_$MODELNAME3 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME3.sh
#ts -L AW_$MODELNAME4 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME4.sh
#ts -L AW_$MODELNAME5 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME5.sh
#ts -L AW_$MODELNAME6 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME6.sh
#ts -L AW_$MODELNAME7 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME7.sh
#ts -L AW_$MODELNAME8 /home/wendt/object_detection/workspace/$DATASETFOLDER/jobs/train_evaluate_export_$MODELNAME8.sh

