#!/bin/sh

# Slurm parameters
#SBATCH -J Train_TF2ODAPI
#SBATCH -N 1 #1 Compute node
#SBATCH --account=p71513
#SBATCH --qos=gpu_gtx1080multi
#SBATCH --partition=gpu_gtx1080multi
#SBATCH --gres gpu:2
#SBATCH --mail-user=alexander.wendt@tuwien.ac.at
#SBATCH --mail-type=BEGIN,END

###
# Functions
###

setup_env()
{
  # Environment preparation
  echo Activate environment
  . ~/eml/init_env_vsc_tf2.sh
}

get_model_name_from_script_name()
{
  # Get file name also for Slurm Script 
  # check if script is started via SLURM or bash
  # if with SLURM: there variable '$SLURM_JOB_ID' will exist
  # `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
  # Source: https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
  echo Slurm Job ID: $SLURM_JOB_ID
  if [ -n $SLURM_JOB_ID ];  then
      # check the original location through scontrol and $SLURM_JOB_ID
      SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
  else
      # otherwise: started with bash. Get the real location.
      SCRIPT_PATH=$(realpath $0)
  fi
  echo Script path: $SCRIPT_PATH

  # getting location of software_name 
  #SHARED_PATH=$(dirname $(dirname $(SCRIPT_PATH)))
  #echo Shared path: $SHARED_PATH
  # separating the software_name from path
  SCRIPT_NAME=$(basename $SCRIPT_PATH)
  echo Base name: $SCRIPT_NAME

  #Extract model name from this filename
  #MYFILENAME=`basename "$0"`
  MYFILENAME=$SCRIPT_NAME
  echo My filename $MYFILENAME

  MODELNAME=`echo $MYFILENAME | sed 's/tf2oda_train_eval_export_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME

}

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

# Get model name
get_model_name_from_script_name

# Environment preparation
setup_env

echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'Using the nodes: '$SLURM_JOB_NODELIST


#echo "Start training of $MODELNAME on VSC" | mail -s "Start training of $MODELNAME" $USEREMAIL


echo "#====================================#"
echo "#Train model"
echo "#====================================#"
echo "model used $MODELNAME"

python $SCRIPTPREFIX/tf2oda_model_main_training.py \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--model_dir=$BASEPATH/models/$MODELNAME

echo "#====================================#"
echo "#Evaluate trained model"
echo "#====================================#"
echo "Evaluate checkpoint performance in tensorboard"

python $SCRIPTPREFIX/tf2oda_evaluate_ckpt_performance.py \
--pipeline_config_path=$BASEPATH/jobs/$MODELNAME.config \
--model_dir=$BASEPATH/models/$MODELNAME \
--checkpoint_dir=$BASEPATH/models/$MODELNAME


echo Read TF Summary from Tensorboard file
python $SCRIPTPREFIX/tf2oda_read_tf_summary.py \
--checkpoint_dir=$BASEPATH/models/$MODELNAME \
--out_dir=result/$MODELNAME/metrics

echo "#====================================#"
echo "#Export inference graph"
echo "#====================================#"
echo "Export model $MODELNAME"
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


#echo "Stop Training of $MODELNAME on EDA02" | mail -s "Training of $MODELNAME finished" $USEREMAIL

echo "#======================================================#"
echo "# Training, evaluation and export of the model completed"
echo "#======================================================#"