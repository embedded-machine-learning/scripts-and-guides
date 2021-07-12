echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
::set SCRIPTPREFIX=..\..\..\
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Convert VOC to Coco
echo #======================================================# 

python C:\Projekte\21_SoC_EML\convert2Yolo\example.py ^
--datasets VOC ^
--img_path C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/images/train ^
--label C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/xmls ^
--convert_output_path C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/yolo_labels ^
--img_type ".jpg" ^
--manifest_path C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/ ^
--cls_list_file C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/labels.txt

::--ann_dir results/%MODELNAME%/validation_for_inference/det_xmls ^
::--labels annotations/labels.txt ^
::--output results/%MODELNAME%/validation_for_inference/coco_pets_detection_annotations.json ^
::--ext xml

::echo if --ann_ids annotations/validation_files.txt  is used, only xmls from the text file are selected, else all files in the folder

::python %SCRIPTPREFIX%\conversion\convert_voc_to_coco.py ^
::--ann_dir annotations/xmls ^
::--ann_ids annotations/validation_files.txt ^
::--labels annotations/labels.txt ^
::--output annotations/coco_pets_validation_annotations.json ^
::--ext xml