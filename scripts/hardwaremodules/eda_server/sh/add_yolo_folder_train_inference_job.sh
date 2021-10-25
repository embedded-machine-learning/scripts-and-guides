#!/bin/bash

# Functions
add_job()
{
  echo "Generate Training Script for $FILENAME"
  cp $CURRENTFOLDER/jobs/$FILENAME $CURRENTFOLDER/$FILENAME
  echo "Add training script to task spooler for $MODELNAME"
  ts -L AW_$MODELNAME $CURRENTFOLDER/$FILENAME
  
  #echo "Generate Inference Script for $MODELNAME"
  #cp tf2oda_inf_eval_saved_model_TEMPLATE.sh tf2oda_inf_eval_saved_model_$MODELNAME.sh
  #echo "Add inference script to task spooler for $MODELNAME"
  #ts -L AW_$MODELNAME $CURRENTFOLDER/tf2oda_inf_eval_saved_model_$MODELNAME.sh
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
MODELSOURCE=jobs/*.sh

echo "#==============================================#"
echo "# CDLEML Tool Add jobs to Task Spooler"
echo "#==============================================#"

echo "Setup task spooler socket."
. ~/init_eda_ts.sh

echo "Current folder: $CURRENTFOLDER"

for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  FILENAME=$(basename $f)
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done

