#!/bin/bash

###
# Functions
###

setup_env_efficientdet()
{
  # Environment preparation
  echo "Activate AutoML EfficientDet environment"
  #call conda activate %PYTHONENV%
  . ./init_env_efficientdet.sh

  # echo "Activate python environment and add python path variables" $PYTHONENV
  #source $PYTHONENV
  
  # Environment preparation
  #echo "Activate environment $PYTHONENV"
  #source /srv/cdl-eml/tf2odapi/venv/$PYTHONENV/bin/activate
  
  echo "Setup task spooler socket."
  . ~/tf2odapi/init_eda_ts.sh
  
  # Set alias python3 if applicable
  alias python=python3
}

setup_env_tf2oda()
{
  echo "Activate AutoML EfficientDet environment"
  . /home/intel-nuc/tf2odapi/init_eda_env.sh
  
  echo "Setup task spooler socket."
  . ~/tf2odapi/init_eda_ts.sh
  
  # Set alias python3 if applicable
  alias python=python3
}

get_model_name()
{
  MYFILENAME=`basename "$0"`
  echo File name: $MYFILENAME
  MODELNAME=`echo $MYFILENAME | sed 's/tf2_effdet_train_export_inf_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
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

train_model()
{
  echo model used $MODELNAME
  rm -rf {./models/$MODELNAME}
  
  python ../../automl/efficientdet/main.py  \
  --mode=train \
  --train_file_pattern=$DATASET_TRAINING/prepared-records/train.record-?????-of-00010 \
  --val_file_pattern=$DATASET_TRAINING/prepared-records/val.record-?????-of-00010 \
  --model_dir=./models/$MODELNAME  \
  --model_name=$MODEL_TYPE \
  --ckpt=pre-trained-models/$MODEL_TYPE \
  --train_batch_size=24 \
  --eval_batch_size=24 --eval_samples=56 \
  --num_examples_per_epoch=$NUMBEREXAMPLES --num_epochs=$NUMBEREPOCHS  \
  --hparams=./config/$CONFIG \
  --val_json_file=$DATASET_TRAINING/annotations/coco_val_annotations.json \
  --strategy=gpus
  
  #Default settings:     
  #  --train_batch_size=64
  #  --eval_batch_size=64
  #  --num_examples_per_epoch=5717 --num_epochs=50
}

export_model()
{
  echo remove exported models folder $MODELNAME if it exists
  rm  -rf ./exported-models/$MODELNAME
 
  echo Export model $MODELNAME
	  
  python ../../automl/efficientdet/model_inspect.py \
  --runmode=saved_model \
  --model_name=$MODEL_TYPE \
  --ckpt_path=./models/$MODELNAME \
  --hparams=./config/$CONFIG \
  --saved_model_dir=./exported-models/$MODELNAME/saved_model \
  --tflite_path=./exported-models/$MODELNAME/saved_model.tflite \
  --min_score_thresh=0.1
  
  echo "rename frozen model with name $MODEL_TYPE\_frozen.pb (TF1) to unified format saved_model_frozen.pb"
  mv exported-models/$MODELNAME/saved_model/$MODEL_TYPE\_frozen.pb exported-models/$MODELNAME/saved_model/saved_model_frozen.pb
  
  echo "Export Saved model to ONNX"
  # Source: https://www.onnxruntime.ai/docs/tutorials/tutorials/tf-get-started.html
  #python -m tf2onnx.convert --saved-model ./exported-models/$MODELNAME/saved_model --output ./exported-models/$MODELNAME/saved_model_unsimplified.onnx --opset 13 --tag serve
  #https://github.com/google/automl/issues/66 

  python -m tf2onnx.convert \
  --saved-model ./exported-models/$MODELNAME/saved_model \
  --output ./exported-models/$MODELNAME/saved_model_unsimplified.onnx \
  --opset 11 \
  --fold_const \
  --target tensorrt \
  --tag serve
  
  echo "Apply ONNX model simplifier"
  
  python -m onnxsim \
  ./exported-models/$MODELNAME/saved_model_unsimplified.onnx \
  ./exported-models/$MODELNAME/saved_model_simplified.onnx \
  3 \
  --input-shape "1,$WIDTH,$HEIGHT,3" \
  --dynamic-input-shape
  
  echo "Export completed"

}

infer_model()
{
  #https://github.com/google/automl/issues/231
  mkdir -p ./results/$MODELNAME/$HARDWARENAME
  
  echo Inference from model
  python $SCRIPTPREFIX/inference_evaluation/tf2effdet_inference_from_saved_model.py \
  --model_path exported-models/$MODELNAME/saved_model/ \
  --image_dir $DATASET_INFERENCE/images/val \
  --detections_out=results/$MODELNAME/$HARDWARENAME/detections.csv \
  --latency_out=results/latency_$HARDWARENAME.csv \
  --min_score=0.5 \
  --latency_runs=100 \
  --model_name=$MODELNAME$ \
  --hardware_name=$HARDWARENAME \
  --index_save_file=./tmp/index.txt
}

evaluate_model()
{
  #echo "#====================================#"
  #echo "# Convert Yolo Detections to Tensorflow Detections CSV Format"
  #echo "#====================================#"
  #echo "Convert Yolo tp TF CSV Format"
  #python $SCRIPTPREFIX/conversion/convert_yolo_to_tfcsv.py \
  #--annotation_dir="results/$MODELNAME/$HARDWARENAME/labels" \
  #--image_dir="$DATASET_INFERENCE/images/val" \
  #--output="results/$MODELNAME/$HARDWARENAME/detections.csv"
  
  
  echo "#====================================#"
  echo "# Convert Detections to Pascal VOC Format"
  echo "#====================================#"
  echo "Convert TF CSV Format similar to voc to Pascal VOC XML"
  python $SCRIPTPREFIX/conversion/convert_tfcsv_to_voc.py \
  --annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
  --output_dir="results/$MODELNAME/$HARDWARENAME/det_xmls" \
  --labelmap_file="$DATASET_INFERENCE/annotations/label_map.pbtxt"


  echo "#====================================#"
  echo "# Convert to Pycoco Tools JSON Format"
  echo "#====================================#"
  echo "Convert TF CSV to Pycoco Tools csv"
  python $SCRIPTPREFIX/conversion/convert_tfcsv_to_pycocodetections.py \
  --annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
  --output_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json"

  echo "#====================================#"
  echo "# Evaluate with Coco Metrics"
  echo "#====================================#"
  echo "coco evaluation"
  python $SCRIPTPREFIX/inference_evaluation/objdet_pycoco_evaluation.py \
  --groundtruth_file="$DATASET_INFERENCE/annotations/coco_val_annotations.json" \
  --detection_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json" \
  --output_file="results/performance_$HARDWARENAME.csv" \
  --model_name=$MODELNAME \
  --hardware_name=$HARDWARENAME \
  --index_save_file="./tmp/index.txt"

  echo "#====================================#"
  echo "# Merge results to one result table"
  echo "#====================================#"
  echo "merge latency and evaluation metrics"
  python $SCRIPTPREFIX/inference_evaluation/merge_results.py \
  --latency_file="results/latency_$HARDWARENAME.csv" \
  --coco_eval_file="results/performance_$HARDWARENAME.csv" \
  --output_file="results/combined_results_$HARDWARENAME.csv"
}

echo "#==============================================#"
echo "# CDLEML Tool TF2 Object Detection API Training"
echo "#==============================================#"

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2_efficientdetd0_512x512_oxfordpets
#PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
DATASET_TRAINING=../../datasets/oxford-pets
DATASET_VALIDATION=../../datasets/oxford-pets
#DATASET_INFERENCE=../../datasets/oxford_pets_reduced
DATASET_INFERENCE=../../datasets/pedestrian_detection_graz_val_only_ss10
CONFIG=oxford-pets_efficientdetd0_640_640_config.yaml
HARDWARENAME=IntelNUC
# Model type: The name is a value used to find the model to used for training. Default is efficientdet-d1. 
MODEL_TYPE=efficientdet-d0
NUMBEREPOCHS=100
NUMBEREXAMPLES=2000
# Set this variable true if the network shall be trained, else only inference shall be performed
TRAINNETWORK=false


# Environment preparation for efficientDet
setup_env_efficientdet

# Get model name
get_model_name

#Extract height and width from model
get_width_and_height

if [ "$TRAINNETWORK" = true ]
then
  #echo "Start training of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Start Train $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL
  
  echo "#====================================#"
  echo "#Train model"
  echo "#====================================#"
  train_model

  echo "#====================================#"
  echo "#Export inference graph to Saved_model"
  echo "#====================================#"
  export_model
  
  #echo "Stop training of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Stop Train $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL
  
else
  echo "No training will take place, only inference"
fi

echo "#====================================#"
echo "#Infer validation images"
echo "#====================================#"

#echo "Start inference of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Start inference $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL

echo "Perform accuracy and latency inference"
infer_model

# Environment preparation for efficientDet
setup_env_tf2oda

echo "Convert values and create evaluation"
evaluate_model


#echo "Inference completed of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Inference complete $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL

echo "#======================================================#"
echo "# Training, evaluation and export of the model completed"
echo "#======================================================#"