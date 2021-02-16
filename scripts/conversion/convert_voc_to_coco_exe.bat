echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_efficientdetd2_768_576_coco17_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pedestrian_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Convert VOC to Coco
echo #======================================================# 

python %SCRIPTPREFIX%\conversion\convert_voc_to_coco.py ^
--ann_dir annotations/xml ^
--ann_ids annotations/train.txt ^
--labels annotations/labels.txt ^
--output annotations/coco_train_annotations.json ^
--ext xml