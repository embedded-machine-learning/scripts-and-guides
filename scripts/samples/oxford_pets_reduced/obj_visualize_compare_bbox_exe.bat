echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

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
echo # Visualize 2 or 3 images with bounding boxes
echo #======================================================# 

python %SCRIPTPREFIX%\inference_evaluation\obj_visualize_compare_bbox.py ^
--labelmap="annotations/%LABELMAP%" ^
--output_dir="results" ^
--image_path1="images/validation/american_bulldog_123.jpg" --annotation_dir1="annotations/xmls" --title1="Ground Truth" ^
--image_path2="images/validation/american_bulldog_123.jpg" --annotation_dir2="results/%MODELNAME%/validation_for_inference/xmls" --title2="SSD-MobileNet" 
::--image_path3="images/validation/TownCentre_frame_2535.jpg" --annotation_dir3="results/tf2oda_efficientdetd2_768_576_coco17_pedestrian/validation_for_inference/xmls" --title3="EfficientDet D2 PETS Only" ^
::--use_three_images