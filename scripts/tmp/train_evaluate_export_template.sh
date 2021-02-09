#!/bin/sh

MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
JOBDIR=star_wars


USERNAME=wendt
PYTHONENV=/home/$USERNAME/object_detection/tf24b/bin/activate
OBJECTDETECTIONFOLDER=/home/$USERNAME/object_detection/models/research/
WORKSPACEFOLDER=/home/$USERNAME/object_detection/workspace/$JOBDIR


#cd ~/object_detection/

echo "Activate python environment and add python path variables"
source $PYTHONENV
cd $OBJECTDETECTIONFOLDER
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

echo "Start training of neural network"
cd $WORKSPACEFOLDER



echo #====================================#
echo # Model evaluation and export
echo #====================================#

echo #====================================#
echo #Train model
echo #====================================#
echo model used $modelname

python 030_model_main_tf2.py --pipeline_config_path=jobs/$modelname.config --model_dir=models/$modelname
echo ## Evaluate during training (in another window or in background)
#python 030_model_main_tf2.py --pipeline_config_path="jobs/%modelname%.config" --model_dir="models/%modelname%" --checkpoint_dir="models/%modelname%"

echo #====================================#
echo #Evaluate trained model
echo #====================================#
python3 040_evaluate_final_performance.py --pipeline_config_path=jobs/$modelname.config --model_dir=models/$modelname --checkpoint_dir=models/$modelname
python3 041_read_tf_summary.py --checkpoint_dir=models/$modelname --out_dir=result/metrics

echo #====================================#
echo #Export inference graph
echo #====================================#
python3 050_exporter_main_v2.py --input_type="image_tensor" --pipeline_config_path=jobs/$modelname.config --trained_checkpoint_dir=models/$modelname --output_directory=exported-models/$modelname

echo #======================================================#
echo #Training, evaluation and export of the model completed
echo #======================================================# 