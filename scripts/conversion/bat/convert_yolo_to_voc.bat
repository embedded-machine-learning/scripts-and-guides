echo #==============================================#
echo # CDLEML Process Data Conversion
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Convert Yolo to Pascal VOC
echo #======================================================# 

python %SCRIPTPREFIX%\conversion\convert_yolo_to_voc.py ^
--annotation_dir "./annotations/yolo_labels" ^
--target_annotation_dir "./annotations/voc_from_yolo_labels" ^
--image_dir "images/train" ^
--class_file "./annotations/labels.txt" ^
--create_empty_images