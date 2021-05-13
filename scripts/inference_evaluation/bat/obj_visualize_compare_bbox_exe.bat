echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_300x300_pets_D100_OVFP16
set PYTHONENV=tf24
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pets_label_map.pbtxt
set HARDWARENAME=Inteli7dp3510

set IMAGENAME=Abyssinian_179.jpg

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Visualize 2 or 3 images with bounding boxes
echo #======================================================# 

python %SCRIPTPREFIX%\inference_evaluation\obj_visualize_compare_bbox.py ^
--labelmap="annotations/%LABELMAP%" ^
--output_dir="results/%MODELNAME%/%HARDWARENAME%" ^
--image_path1="images/validation/%IMAGENAME%" --annotation_dir1="annotations/xmls" --title1="Ground Truth" ^
--image_path2="images/validation/%IMAGENAME%" --annotation_dir2="results/%MODELNAME%/%HARDWARENAME%/det_xmls" --title2="SSD-MobileNetFP16-OpenVino" ^
--color_gt
::--image_path3="images/validation/TownCentre_frame_2535.jpg" --annotation_dir3="results/tf2oda_efficientdetd2_768_576_coco17_pedestrian/validation_for_inference/xmls" --title3="EfficientDet D2 PETS Only" ^
::--use_three_images
