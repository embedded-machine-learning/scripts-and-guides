echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo Execute this file in the base folder of your project

:: Constants Definition
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set MODELNAMESHORT=MobNetV2_300x300_D100
set HARDWARENAME=CPU_Intel_i5
set PYTHONENV=tf24
::set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set SCRIPTPREFIX=..\..\..
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo #Visualize Detections of Images
echo #====================================#

echo WARNING: Script does not work if only one detectionbox was found

python %SCRIPTPREFIX%\inference_evaluation\visualize_tf2oda_csv_detections.py ^
--image_dir="images/validation" ^
--labelmap="annotations/%LABELMAP%" ^
--detections_file="results/%MODELNAME%/validation_for_inference/detections.csv" ^
--min_score=0.5 ^
--output_dir="results/%MODELNAME%/validation_for_inference"


