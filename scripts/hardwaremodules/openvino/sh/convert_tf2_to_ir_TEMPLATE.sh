#!/bin/bash

###
# Functions
###

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/convert_tf2_to_ir_//' | sed 's/.sh//'`
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

convert_to_ir()
{
  echo Apply to model %MODELNAME% with precision %PRECISION%


  echo "Define API config file"
  APIFILEEFF=$SCRIPTPREFIX/hardwaremodules/openvino/openvino_conversion_config/efficient_det_support_api_v2.4.json
  APIFILESSD=$SCRIPTPREFIX/hardwaremodules/openvino/openvino_conversion_config/ssd_support_api_v2.4.json
  APIFILE=ERROR

  if not x%MODELNAME:ssd=%==x%MODELNAME% (set APIFILE=%APIFILESSD%)
  if not x%MODELNAME:effi=%==x%MODELNAME% (set APIFILE=%APIFILEEFF%)
  echo Use this API file: $APIFILE
  
  echo "Start conversion"
  python3 $OPENVINOINSTALLDIR/model-optimizer/mo_tf.py \
  --saved_model_dir="exported-models/$MODELNAME/saved_model" \
  --tensorflow_object_detection_api_pipeline_config=exported-models/$MODELNAME/pipeline.config \
  --transformations_config=$APIFILE \
  --reverse_input_channels \
  --data_type $PRECISION \
  --output_dir=exported-models-openvino/$MODELNAME\_OV$PRECISION

  echo "Conversion finished"
}

#convert_to_trt()
#{
#  python3 $SCRIPTPREFIX/hardwaremodules/nvidia/convert_tf2_to_trt.py \
#  --tensorflow_model=exported-models/$MODELNAME/saved_model \
#  --batch_size=8 \
#  --image_size="[$height, $width]" \
#  --precision=$PRECISION \
#  --dtype=uint8 \
#  --data_dir=./images/validation \
#  --output_dir=./exported-models-trt/$MODELNAME\_TRT$PRECISION
#}


###
# Main body of script starts here
###

echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

# Constant Definition
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdet_512x384_pedestrian_D0_LR02
PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
MODELSOURCE=jobs/*.config
HARDWARENAME=IntelNUC
LABELMAP=pedestrian_label_map.pbtxt

#Extract model name from this filename
get_model_name

#Setup environment
#setup_env

#echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL

#echo "Setup task spooler socket."
#. ~/init_eda_ts.sh


echo Apply to model $MODELNAME
get_width_and_height

#Get image resolution from model name

alias python=python3

echo #====================================#
echo # Convert TF2 Model to OpenVino Intermediate Representation
echo #====================================#

#PRECISION=INT8

PRECISIONLIST="FP16 FP32"

for PRECISION in $PRECISIONLIST
do
  #echo "$f"
  #MODELNAME=`basename ${f%%.*}`
  echo $PRECISION
  convert_to_ir
  
done
