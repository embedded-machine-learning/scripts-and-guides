#!/bin/bash

#BASESCRIPT=tf2oda_train_eval_export
#BASESCRIPT=tf2_inf_eval_saved_model

# Functions
add_job()
{
  echo "Generate Training Script for $MODELNAME"
  cp $SCRIPTBASENAME\_TEMPLATE.sh $SCRIPTBASENAME\_$MODELNAME.sh
  echo "Add task spooler jobs for $MODELNAME to the task spooler"
  echo "Shell script tf2_inf_eval_saved_model_$MODELNAME.sh"
  ts -L AW_$MODELNAME $CURRENTFOLDER/$SCRIPTBASENAME\_$MODELNAME.sh
}

###
# Main body of script starts here
###

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
PYTHONENV=tf24
#SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training
CURRENTFOLDER=`pwd`
#MODELSOURCE=jobs/*.config
MODELSOURCE=exported-models/*
#MODELSOURCE=temp/exported-models-temp/*
SCRIPTBASENAME=convert_tf2_to_ir

echo "Setup task spooler socket."
. ~/tf2odapi/init_eda_ts.sh


for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  echo $MODELNAME
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done

