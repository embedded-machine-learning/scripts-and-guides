#!/bin/bash

###
# Functions
###

setup_env()
{
  # Environment preparation
  echo Activate environment $PYTHONENV
  #call conda activate %PYTHONENV%
  #Environment is put directly in the nuc home folder
  . ~/tf2odapi/init_eda_env.sh
}

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/tf2_inf_eval_saved_model_//' | sed 's/.sh//'`
  echo Selected model based on folder name: $MODELNAME
}

get_width_and_height()
{
  elements=(${MODELNAME//_/ })
  #$(echo $MODELNAME | tr "_" "\n")
  #echo $elements
  resolution=${elements[2]}
  res_split=(${resolution//x/ })
  height=${res_split[0]}
  width=${res_split[1]}

  echo batch processing height=$height and width=$width

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
PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
MODELSOURCE=jobs/*.config
HARDWARENAME=IntelNUC
DATASET=../../datasets/pedestrian_detection_graz_val_only_ss10
#DATASET=../../datasets/pedestrian_detection_graz_val_only_debug
LABELMAP=label_map.pbtxt

#Extract model name from this filename
get_model_name

#Extract height and width from model
get_width_and_height

#Setup environment
setup_env

#echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL

#echo "Setup task spooler socket."
. ~/tf2odapi/init_eda_ts.sh


echo Apply to model $MODELNAME

echo #====================================#
echo # Infer Images from Known Model
echo #====================================#

echo Inference from model 
python $SCRIPTPREFIX/inference_evaluation/tf2oda_inference_from_saved_model.py \
--model_path "exported-models/$MODELNAME/saved_model/" \
--image_dir "$DATASET/images/val" \
--detections_out="results/$MODELNAME/$HARDWARENAME/detections.csv" \
--latency_out="results/latency_$HARDWARENAME.csv" \
--min_score=0.5 \
--model_name=$MODELNAME \
--hardware_name=$HARDWARENAME \
--batch_size=1 \
--index_save_file="./tmp/index.txt"

#--image_size="[$height, $width]" Optional to use if another size as provided in the model name is used
#--batch_size: Default=1
#--model_short_name=%MODELNAMESHORT% unused because the name is created in the csv file


#echo #====================================#
#echo # Convert Detections to Pascal VOC Format
#echo #====================================#
#echo "Convert TF CSV Format similar to voc to Pascal VOC XML"
#python $SCRIPTPREFIX/conversion/convert_tfcsv_to_voc.py \
#--annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
#--output_dir="results/$MODELNAME/$HARDWARENAME/det_xmls" \
#--labelmap_file="$DATASET/annotations/$LABELMAP"


echo #====================================#
echo # Convert to Pycoco Tools JSON Format
echo #====================================#
echo "Convert TF CSV to Pycoco Tools csv"
python $SCRIPTPREFIX/conversion/convert_tfcsv_to_pycocodetections.py \
--annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
--output_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json"

echo #====================================#
echo # Evaluate with Coco Metrics
echo #====================================#

python $SCRIPTPREFIX/inference_evaluation/objdet_pycoco_evaluation.py \
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
