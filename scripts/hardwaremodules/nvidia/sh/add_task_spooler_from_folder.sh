#!/bin/bash

#BASESCRIPT=tf2oda_train_eval_export
BASESCRIPT=tf2_inf_eval_saved_model

# Functions
add_job()
{
  echo "Generate Training Script for $MODELNAME"
  cp convert_tf2_to_trt_TEMPLATE.sh convert_tf2_to_trt_$MODELNAME.sh
  echo "Add task spooler jobs for $MODELNAME to the task spooler"
  echo "Shell script tf2_inf_eval_saved_model_$MODELNAME.sh"
  tsp -L AW_$MODELNAME $CURRENTFOLDER/convert_tf2_to_trt_$MODELNAME.sh
}

###
# Main body of script starts here
###

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
PYTHONENV=tf24
BASEPATH=.
SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training
CURRENTFOLDER=`pwd`
#MODELSOURCE=jobs/*.config
MODELSOURCE=exported-models/*
#MODELSOURCE=temp/*

echo "Setup task spooler socket."
. /media/cdleml/128GB/Users/awendt/init_eda_ts.sh


for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  echo $MODELNAME
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done

