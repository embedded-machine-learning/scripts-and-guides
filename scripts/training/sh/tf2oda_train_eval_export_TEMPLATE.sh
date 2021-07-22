#!/bin/sh

echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=~/tf2odapi/scripts-and-guides/scripts/training

#Extract model name from this filename
MYFILENAME=`basename "$0"`
MODELNAME=`echo $MYFILENAME | sed 's/tf2oda_train_eval_export_//' | sed 's/.sh//'`
echo Selected model: $MODELNAME


# Environment preparation
echo Activate environment $PYTHONENV
#call conda activate %PYTHONENV%
. ~/init_eda_env.sh

# echo "Activate python environment and add python path variables" $PYTHONENV
#source $PYTHONENV

echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL


echo #====================================#
echo #Train model
echo #====================================#
echo model used $MODELNAME

python $SCRIPTPREFIX/tf2oda_model_main_training.py \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--model_dir=$BASEPATH/models/$MODELNAME \
--time_measurement_path=results/$MODELNAME/metrics/training_time.txt

echo #====================================#
echo #Evaluate trained model
echo #====================================#
echo Evaluate checkpoint performance in tensorboard

python $SCRIPTPREFIX/tf2oda_evaluate_ckpt_performance.py \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--model_dir=$BASEPATH/models/$MODELNAME \
--checkpoint_dir=$BASEPATH/models/$MODELNAME


echo Read TF Summary from Tensorboard file
python $SCRIPTPREFIX/tf2oda_read_tf_summary.py \
--checkpoint_dir=$BASEPATH/models/$MODELNAME \
--out_dir=results/$MODELNAME/metrics

echo #====================================#
echo #Export inference graph
echo #====================================#
echo Export model %modelname%
python $SCRIPTPREFIX/tf2oda_export_savedmodel.py \
--input_type="image_tensor" \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--trained_checkpoint_dir=$BASEPATH/models/$MODELNAME \
--output_directory=exported-models/$MODELNAME

echo "Export Saved model to ONNX and apply ONNX model simplifier"
# Source: https://www.onnxruntime.ai/docs/tutorials/tutorials/tf-get-started.html
python -m tf2onnx.convert --saved-model ./exported-models/$MODELNAME/saved_model --output ./exported-models/$MODELNAME/saved_model_unsimplified.onnx --opset 13 --tag serve
#python -m onnxsim ./exported-models/tf2oda_ssdmobilenetv2_320x320_peddet20/saved_model_unsimplified.onnx ./exported-models/tf2oda_ssdmobilenetv2_320x320_peddet20/saved_model.onnx


echo "Stop Training of $MODELNAME on EDA02" | mail -s "Training of $MODELNAME finished" $USEREMAIL

echo #======================================================#
echo # Training, evaluation and export of the model completed
echo #======================================================# 