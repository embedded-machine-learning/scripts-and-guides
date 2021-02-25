echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo execute in ..\..\scripts-and-guides\scripts\training\samples\oxford_pets_reduced

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
::set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set SCRIPTPREFIX=..\..\..\
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Convert VOC to Coco
echo #======================================================# 

echo if --ann_ids annotations/validation_files.txt  is used, only xmls from the text file are selected, else all files in the folder
echo WARNING: In case coco metric validation files is created, no orphan XML must exists, else they are considered in the metric although no images are available

python %SCRIPTPREFIX%\conversion\convert_voc_to_coco.py ^
--ann_dir results/%MODELNAME%/validation_for_inference/det_xmls ^
--labels annotations/labels.txt ^
--output results/%MODELNAME%/validation_for_inference/coco_pets_detection_annotations.json ^
--ext xml

