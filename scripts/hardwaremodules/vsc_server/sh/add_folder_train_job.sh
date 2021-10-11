#!/bin/bash

# Functions
add_job()
{
  echo "Generate Training Script for $MODELNAME"
  cp tf2oda_train_eval_export_TEMPLATE.sh tf2oda_train_eval_export_$MODELNAME.sh
  echo "Add task spooler jobs for $MODELNAME to slurm"
  #ts -L AW_$MODELNAME $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME.sh
  sbatch $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME.sh
}

###
# Main body of script starts here
###

# Constant Definition
#USERNAME=wendt
#USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
#PYTHONENV=tf24
#BASEPATH=.
#SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training
CURRENTFOLDER=`pwd`
MODELSOURCE=jobs/*.config


echo "#==============================================#"
echo "# CDLEML Tool Add jobs to VSC Slurm"
echo "#==============================================#"

#echo "Setup task spooler socket."
#. ~/init_eda_ts.sh

#bash $CURRENTFOLDER/tf2oda_train_eval_export_tf2oda_ssdmobilenetv2_300x300_pets_s1000.sh

for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done

