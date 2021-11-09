#!/bin/bash
#
echo '========================================'
echo ' Installing CUDA environment (Conda)'
echo '========================================'
#
echo 'Load Anaconda and GCC modules'
module purge
module load anaconda3/5.3.0 gcc/9.1.0
#
echo 'Activate Anaconda hook'
source /opt/sw/x86_64/glibc-2.17/ivybridge-ep/anaconda3/5.3.0/etc/profile.d/conda.sh
#
echo 'Create and activate new conda env named cuda'
conda deactivate
conda create -n cuda -y python=3.7 cuda=11.4.2 cudnn=8.0.4 -c nvidia
conda activate cuda
conda install -y opencv=4.5.3 -c conda-forge
#
echo 'Installing basic pip packages'
python -m pip install --upgrade pip
# install numpy version 1.19.4 because there is a bug in 1.19.5 which sometimes makes problems when installing matplotlib
pip install --upgrade setuptools numpy==1.19.4 cython wheel

echo 'Installing pip packages for eml tools'
# install other necessary libraries for eml tools
pip install lxml xmltodict tdqm beautifulsoup4

#
echo 'Installing requirements from .txt file'
# installing packages needed for the prep-kaist.py
pip install -r requirements.txt
#

# If problems, upgrade numpy
pip install numpy --upgrade

echo 'Done!'
