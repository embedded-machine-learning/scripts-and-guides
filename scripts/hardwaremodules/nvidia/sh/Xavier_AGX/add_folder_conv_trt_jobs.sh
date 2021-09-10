#!/bin/bash

#BASESCRIPT=tf2oda_train_eval_export
BASESCRIPT=tf2_inf_eval_saved_model

# Functions
add_job()
{
  echo "Generate Training Script for $MODELNAME"
  echo "Copy convert_tf2_to_trt_TEMPLATE.sh to convert_tf2_to_trt_$MODELNAME.sh"
  cp convert_tf2_to_trt_TEMPLATE.sh convert_tf2_to_trt_$MODELNAME.sh
  
  #echo "Copy tf2_inf_eval_saved_model_TEMPLATE.sh to tf2_inf_eval_saved_model_$MODELNAME.sh"
  #cp tf2_inf_eval_saved_model_TEMPLATE.sh tf2_inf_eval_saved_model_$MODELNAME.sh
  
  #echo "Copy tf2_inf_eval_saved_model_trt_TEMPLATE.sh to tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh"
  #cp tf2_inf_eval_saved_model_trt_TEMPLATE.sh tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh
  
  #echo "Copy tf2_inf_eval_saved_model_trt_TEMPLATE.sh to tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh"
  #cp tf2_inf_eval_saved_model_trt_TEMPLATE.sh tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh
  
  #echo "Copy tf2_inf_eval_saved_model_trt_TEMPLATE.sh to tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh"
  #cp tf2_inf_eval_saved_model_trt_TEMPLATE.sh tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh
  
  echo "Add task spooler jobs for $MODELNAME to the task spooler"
  echo "Add shell script convert_tf2_to_trt_$MODELNAME.sh"
  tsp -L AW_conv_$MODELNAME $CURRENTFOLDER/convert_tf2_to_trt_$MODELNAME.sh
  #echo "Add shell script tf2_inf_eval_saved_model_$MODELNAME.sh"
  #tsp -L AW_inf_orig_$MODELNAME $CURRENTFOLDER/tf2_inf_eval_saved_model_$MODELNAME.sh
  #echo "Add shell script tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh"
  #tsp -L AW_inf_FP32_$MODELNAME $CURRENTFOLDER/tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh
  #echo "Add shell script tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh"
  #tsp -L AW_inf_FP16_$MODELNAME $CURRENTFOLDER/tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh
  #echo "Add shell script tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh"
  #tsp -L AW_inf_INT8_$MODELNAME $CURRENTFOLDER/tf2_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh
  
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
. /media/cdleml/128GB/Users/awendt/init_xavier_ts.sh

echo "This file converts a saved_model.pb from exported-models into a TRT models for FP32, FP16 and INT8. Then it executes inference on all 4 models and saved the models in results."

for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  echo $MODELNAME
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done

