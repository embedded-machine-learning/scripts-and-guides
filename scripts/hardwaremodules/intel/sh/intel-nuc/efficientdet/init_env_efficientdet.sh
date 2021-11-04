#!/bin/bash

echo WARNING: Start script with . /[PATH_TO_SCRIPT], not with ./
echo WARNING: Set variable ENVROOT to your root for environment and models.

echo "Setup Task Spooler"
#. /srv/cdl-eml/tf2odapi/init_eda_ts.sh
. /home/intel-nuc/tf2odapi/init_eda_ts.sh

echo Activate TF2 AutoML EfficientDet Environment and exclude TF2ODA

TF2ROOT=`pwd`
ENVROOT=/home/intel-nuc/tf2odapi

#source $ENVROOT/venv/effdet_py38/bin/activate
source $ENVROOT/tf24/bin/activate
cd $ENVROOT/automl/efficientdet
export PYTHONPATH=`pwd`
echo New python path $PYTHONPATH

cd $TF2ROOT

echo Activation complete
