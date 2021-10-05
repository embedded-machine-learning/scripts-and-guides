#!/bin/sh

echo "#==============================================#"
echo "# CDLEML Tool TF2 Object Detection API Training"
echo "#==============================================#"

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts/training


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

echo "Start training of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Start Train $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL


echo "#====================================#"
echo "#Train model"
echo "#====================================#"
echo model used $MODELNAME

python $SCRIPTPREFIX/tf2oda_model_main_training.py \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--model_dir=$BASEPATH/models/$MODELNAME

echo "#====================================#"
echo "#Evaluate trained model"
echo "#====================================#"
echo Evaluate checkpoint performance in tensorboard

python $SCRIPTPREFIX/tf2oda_evaluate_ckpt_performance.py \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--model_dir=$BASEPATH/models/$MODELNAME \
--checkpoint_dir=$BASEPATH/models/$MODELNAME


echo Read TF Summary from Tensorboard file
python $SCRIPTPREFIX/tf2oda_read_tf_summary.py \
--checkpoint_dir=$BASEPATH/models/$MODELNAME \
--out_dir=results/$MODELNAME/metrics

echo "#====================================#"
echo "#Export inference graph"
echo "#====================================#"
echo Export model $MODELNAME
python $SCRIPTPREFIX/tf2oda_export_savedmodel.py \
--input_type="image_tensor" \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--trained_checkpoint_dir=$BASEPATH/models/$MODELNAME \
--output_directory=exported-models/$MODELNAME

echo "#====================================#"
echo "#Copy Exported Graph to Pickup folder"
echo "#====================================#"
mkdir tmp
cp -ar exported-models/$MODELNAME tmp


echo "Stop training of $MODELNAME on EDA02 $(date +"%Y%m%d %T")" | mail -s "Stop Train $MODELNAME EDA02 $(date +"%Y%m%d %T")" $USEREMAIL

echo "#======================================================#"
echo "# Training, evaluation and export of the model completed"
echo "#======================================================#"