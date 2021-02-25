#!/bin/bash

echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts
PROJECTNAME=star_wars_detection

# Environment preparation
echo Activate environment $PYTHONENV
#call conda activate %PYTHONENV%
. ~/init_eda_env.sh


echo Compress results

mkdir tmp

python $SCRIPTPREFIX/training/zip_tool.py \
--items="results, exported-models" \
--out=./tmp/$PROJECTNAME\_results.zip