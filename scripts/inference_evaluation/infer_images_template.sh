#!/bin/sh

modelname=ssd_mobilenet_v2_300x300_coco17_starwars

echo #====================================#
echo #Infer new images
echo #====================================#

python3 060_inference_from_saved_model_tf2.py -p exported-models/$modelname/saved_model/ -i images/inference -l annotations/sw_label_map.pbtxt --output_dir=result -r