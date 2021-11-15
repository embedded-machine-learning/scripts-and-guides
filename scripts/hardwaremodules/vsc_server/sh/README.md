# Guide how to setup Machine Learning with Tensorflow 2.4.1 on VSC3

When you connect to VSC, you only connect to the Login nodes. To use GPU, you have to either send jobs via Slurm to the GPU nodes or login in interactive mode to a login node.

Credits to Thomas Kotrba for the support.

## Setup interactive mode
The interactive mode is used to try out thing, e.g. loading necessary modules and setting up your environment.

Guide how to login: https://wiki.vsc.ac.at/doku.php?id=doku:slurm_interactive

You can login to the following nodes
Your jobs can run with the following account(s) and quality of service (QOS):
default_account:              p71513
        account:              p71513

    default_qos:       vsc3plus_0064
            qos:          devel_0064
                         gpu_a40dual
                      gpu_gtx1080amd
                    gpu_gtx1080multi
                   gpu_gtx1080single
                            gpu_k20m
                       gpu_rtx2080ti
                            gpu_v100
                             jupyter
                         normal_binf
                       vsc3plus_0064
                       vsc3plus_0256

### Steps to connect to a GPU node in interactive mode:
1. Allocate ressources, example: 
salloc -p gpu_gtx1080multi --qos gpu_gtx1080multi --gres=gpu:2
a.	gpu_gtx1080multi is the qos type
b.	--gres=gpu:2 means 2 gpu cores
2. Check the name of the assigned node to connect to. You are still on the login node: 
srun hostname 
and you get the names of the available nodes, e.g. n371-004.vsc3.prov
3. Connect through ssh 
ssh n371-004
to get to the actual node with the GPUs
4. Check your current queue on that node
squeue --account [ACCOUNT NAME]

Note: Quit interactive session with exit and exit. It shall look like this:
salloc: Relinquishing job allocation 10016841
salloc: Job allocation 10016841 has been revoked.
sqos: see available ressources. with -aac you see available nodes for your account

### Setup Tensorflow 2.4.1 Environment
Setup Environment:
1)	Load NVIDIA driver, Cuda and Anaconda
module load anaconda3/5.3.0 cuda/11.0.2 nvidia/1.0
2)	Check, which GPU resources are available
nvidia-smi
 
3)	Setup conda environment with Python 3.7 for Tensorflow 2.4.1 (3.8 does not work with anaconda 5.3.0) and install the following libraries: 
a.	pip install tensorflow==2.4.1
b.	conda install -c anaconda cudatoolkit=11.0
c.	conda install -c conda-forge cudnn=8

To get TF running, you need both CUDA and CUDNN. Conda will install Cudatoolkit 11.3 at the installation of CUDNN 8.2, however, testing the system seems to work as probably the cuda/11.0.2 is active in the system provided by VSC.
Links: https://anaconda.org/conda-forge/cudnn (use for cuDNN, https://anaconda.org/anaconda/cudatoolkit/files (use for CUDA)
The actual installed version: https://anaconda.org/anaconda/cudnn/8.2.1/download/linux-64/cudnn-8.2.1-cuda11.3_0.tar.bz2 

4)	Test Tensorflow implementation 
a.	Python
b.	import tensorflow as tf
c.	hello = tf.constant('Hello, TensorFlow!')
d.	tf.config.list_physical_devices('GPU')

Conda commands:
List environemnts: conda env list
Create environment: conda create --name [project-env] python=3.7
Remove environment: conda env remove --name [project-env]
Activate environment: conda activate [project-env]

I case the GPU devices are not visible, execute
export CUDA_VISIBLE_DEVICES=0,1

## Tensorflow 2 Object Detection API and EML Tools Testing
Setup Tensorflow 2 Object Detection API: 
1)	see https://github.com/embedded-machine-learning/scripts-and-guides/blob/main/guides/setup_tf2_object_detection_api.md 
2)	test training in /home/lv71513/awendt/eml/demonstration_projects/oxford_pets_detection_training
3)	Copy your data to /binfl/lv71513/awendt/datasets  and don’t put them into the home directory as it gets full

If everything was setup correctly, a model was trained, evaluated and exported.


## Setup a Slurm Job With the EML Tools
Guide how to setup jobs for VSC: https://wiki.vsc.ac.at/doku.php?id=doku:slurm

Slurm Commands
Add job: sbatch [JOB SCRIPT].sh
Get own jobs: squeue -u `whoami`
Get info of a certain Job id: scontrol show job [JOB ID]
Cancel job: scancel  [JOB ID]

Environment Setup in Slurm for VSC3
The following script is an environment setup script that is executed in every job script. Yellow text are for necessary lines for usage of conda.

```
#!/bin/bash

# put in home
echo "WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./"
echo "WARNING: Set variable ENVROOT to your root for environment and TF2ODA models."

# Load VSC modules
module purge #clear
module load anaconda3/5.3.0 cuda/11.0.2 nvidia/1.0 #Load models that we need

echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'Using the nodes: '$SLURM_JOB_NODELIST

echo "Activate TF2 Object Detection API Python Environment"
TF2ROOT=`pwd`
ENVROOT=/home/lv71513/awendt/eml

source /opt/sw/x86_64/glibc-2.17/ivybridge-ep/anaconda3/5.3.0/etc/profile.d/conda.sh
conda activate tf24

# For TF2ODA special path
cd $ENVROOT/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo New python path $PYTHONPATH

cd $TF2ROOT
echo "Activation complete"
```

Job Script Setup for a Slurm job on VSC3
The following script is used to setup a queued job on VSC3. Yellow text is necessary to configure the Slurm job. The rest is for a general TF2ODA job. The script is added to the node with squeue [SCRIPT_NAME].sh.

```
#!/bin/sh

# Slurm parameters
#SBATCH -J Train_TF2ODAPI
#SBATCH -N 1 #1 Compute node
#SBATCH --account=p71513
#SBATCH --qos=gpu_gtx1080multi
#SBATCH --partition=gpu_gtx1080multi
#SBATCH --gres gpu:2 # Use 2 GPUs on the Node
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

```

