echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
PYTHONENV=tf24
BASEPATH=.
SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training

# Environment preparation
echo Activate environment $PYTHONENV
#call conda activate %PYTHONENV%
. ~/init_eda_env.sh

# echo "Activate python environment and add python path variables" $PYTHONENV
#source $PYTHONENV

echo "Start training of $MODELNAME" | mail -s "Start training of $MODELNAME" $USEREMAIL


echo #====================================#
echo #Train model
echo #====================================#
echo model used $MODELNAME

python $SCRIPTPREFIX/tf2oda_model_main_training.py /
--pipeline_config_path="$BASEPATH/jobs/$MODELNAME.config" /
--model_dir="$BASEPATH/models/$MODELNAME"

echo #====================================#
echo #Evaluate trained model
echo #====================================#
echo Evaluate checkpoint performance

python $SCRIPTPREFIX/tf2oda_evaluate_ckpt_performance.py /
--pipeline_config_path="jobs/$modelname.config" /
--model_dir="$BASEPATH/models/$MODELNAME" /
--checkpoint_dir="$BASEPATH/models/$MODELNAME$"

echo Read TF Summary from Tensorboard file

python $SCRIPTPREFIX/tf2oda_read_tf_summary.py /
--checkpoint_dir="$BASEPATH/models/$MODELNAME" /
--out_dir="result/$MODELNAME/metrics"

echo #====================================#
echo #Export inference graph
echo #====================================#
echo Export model %modelname%
python $SCRIPTPREFIX/tf2oda_export_savedmodel.py /
--input_type="image_tensor" /
--pipeline_config_path="$BASEPATH/jobs/$MODELNAME.config" /
--trained_checkpoint_dir="$BASEPATH/models/$MODELNAME" /
--output_directory="exported-models/$MODELNAME"

::echo "Training of $MODELNAME finished" | mail -s "Training of $MODELNAME finished" $USEREMAIL

echo #======================================================#
echo # Training, evaluation and export of the model completed
echo #======================================================# 