#!/bin/bash

#echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

echo "=== Create Dynamic Links to Dataset ==="

# Dataset has to be an abolute folder name
DATASET=/home/intel-nuc/tf2odapi/datasets/pedestrian_detection_graz_val_only

echo "DATSET: $DATASET"
rm -r ./dataset
mkdir ./dataset

echo "Link annotations"
ln -s $DATASET/annotations ./dataset

echo "Link images"
ln -s $DATASET/images ./dataset

ls ./dataset

echo "Dynamic Links created"
