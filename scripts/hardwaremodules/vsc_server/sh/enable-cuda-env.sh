#!/bin/bash
#
echo '========================================'
echo ' Enabling CUDA environment (Conda)'
echo '========================================'
echo ' Please run this script on a GPU Node!'
#
echo 'Load Anaconda and GCC modules'
module purge
module load anaconda3/5.3.0 gcc/9.1.0
#
echo 'Activate Anaconda hook'
source /opt/sw/x86_64/glibc-2.17/ivybridge-ep/anaconda3/5.3.0/etc/profile.d/conda.sh
#
echo 'Activate conda env cuda'
conda deactivate
conda activate cuda
#
echo 'Adding path to env modules'
export PATH=$PATH:~/.conda/envs/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cuda/lib 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/cuda/lib/stubs 
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:~/.conda/envs/cuda/lib/pkgconfig # for opencv
#
echo 'Making the GPUs visible'
export CUDA_VISIBLE_DEVICES=0,1
#
echo 'Done!'
