#!/bin/bash

###
# Functions
###

#add_job()
#{
#  echo "Generate Training Script for $MODELNAME"
#  cp tf2oda_train_eval_export_TEMPLATE.sh tf2oda_train_eval_export_$MODELNAME.sh
#  echo "Add task spooler jobs for $MODELNAME to the task spooler"
#  ts -L AW_$MODELNAME $CURRENTFOLDER/tf2oda_train_eval_export_$MODELNAME.sh
#}

setup_env()
{
  # Environment preparation
  echo Activate environment
  #call conda activate %PYTHONENV%
  . ~/init_eda_env.sh

  # echo "Activate python environment and add python path variables" $PYTHONENV
  #source $PYTHONENV

}

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/tf2oda_inf_eval_saved_model_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
}


###
# Main body of script starts here
###

echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

# Constant Definition
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
#PYTHONENV=tf24
#BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
#DATASET=../../datasets/pedestrian_detection_graz_val_only
DATASET=../../datasets/pedestrian_detection_graz_val_only_ss10
#MODELSOURCE=jobs/*.config
HARDWARENAME=TeslaV100
LABELMAP=label_map.pbtxt

#Extract model name from this filename
get_model_name

#Setup environment
setup_env

echo "Start inference of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Start Inference $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL

#echo "Setup task spooler socket."
#. ~/init_eda_ts.sh


echo Apply to model $MODELNAME

echo #====================================#
echo # Infer Images from Known Model
echo #====================================#

echo Inference from model 
python3 $SCRIPTPREFIX/inference_evaluation/tf2oda_inference_from_saved_model.py \
--model_path "exported-models/$MODELNAME/saved_model/" \
--image_dir "$DATASET/images/val" \
--labelmap "$DATASET/annotations/$LABELMAP" \
--detections_out="results/$MODELNAME/$HARDWARENAME/detections.csv" \
--latency_out="results/latency_$HARDWARENAME.csv" \
--min_score=0.5 \
--model_name=$MODELNAME \
--hardware_name=$HARDWARENAME \
--index_save_file="./tmp/index.txt"

#--model_short_name=%MODELNAMESHORT% unused because the name is created in the csv file


echo #====================================#
echo # Convert Detections to Pascal VOC Format
echo #====================================#
echo Convert TF CSV Format similar to voc to Pascal VOC XML
python $SCRIPTPREFIX/conversion/convert_tfcsv_to_voc.py \
--annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
--output_dir="results/$MODELNAME/$HARDWARENAME/det_xmls" \
--labelmap_file="$DATASET/annotations/$LABELMAP"


echo #====================================#
echo # Convert to Pycoco Tools JSON Format
echo #====================================#
echo Convert TF CSV to Pycoco Tools csv
python3 $SCRIPTPREFIX/conversion/convert_tfcsv_to_pycocodetections.py \
--annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
--output_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json"

echo #====================================#
echo # Evaluate with Coco Metrics
echo #====================================#
echo coco evaluation
python3 $SCRIPTPREFIX/inference_evaluation/objdet_pycoco_evaluation.py \
--groundtruth_file="$DATASET/annotations/coco_val_annotations.json" \
--detection_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json" \
--output_file="results/performance_$HARDWARENAME.csv" \
--model_name=$MODELNAME \
--hardware_name=$HARDWARENAME \
--index_save_file="./tmp/index.txt"

echo #====================================#
echo # Merge results to one result table
echo #====================================#
echo merge latency and evaluation metrics
python3 $SCRIPTPREFIX/inference_evaluation/merge_results.py \
--latency_file="results/latency_$HARDWARENAME.csv" \
--coco_eval_file="results/performance_$HARDWARENAME.csv" \
--output_file="results/combined_results_$HARDWARENAME.csv"

echo "Stop inference of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Stop Inference $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL