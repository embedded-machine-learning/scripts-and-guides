#!/bin/sh

MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
USERNAME=wendt
JOBDIR=eda01_validation_tf2_star_wars


#cd ..

cd ~/object_detection/

echo "Activate python environment and add python path variables"
source /home/$USERNAME/object_detection/tf24b/bin/activate
cd /home/$USERNAME/object_detection/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

echo "Start training of neural network"
cd /home/$USERNAME/object_detection/workspace/$JOBDIR

echo "#====================================#"
echo "#Train model"
echo "#====================================#"
echo "model used $MODELNAME"

python 030_model_main_tf2.py --pipeline_config_path=jobs/$MODELNAME.config --model_dir=models/$MODELNAME
#echo ## Evaluate during training (in another window or in background)
#python 030_model_main_tf2.py --pipeline_config_path="jobs/%MODELNAME%.config" --model_dir="models/%MODELNAME%" --checkpoint_dir="models/%MODELNAME%"

echo "#====================================#"
echo "#Evaluate trained model"
echo "#====================================#"
python 040_evaluate_final_performance.py --pipeline_config_path=jobs/$MODELNAME.config --model_dir=models/$MODELNAME --checkpoint_dir=models/$MODELNAME
python 041_read_tf_summary.py --checkpoint_dir=models/$MODELNAME --out_dir=result/metrics

echo "#====================================#"
echo "#Export inference graph"
echo "#====================================#"
python 050_exporter_main_v2.py --input_type="image_tensor" --pipeline_config_path=jobs/$MODELNAME.config --trained_checkpoint_dir=models/$MODELNAME --output_directory=exported-models/$MODELNAME

echo "#======================================================#"
echo "#Training, evaluation and export of the model completed"
echo "#======================================================#" 