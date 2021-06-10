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
echo #Infer new images
echo #====================================#

python %SCRIPTPREFIX%\inference_evaluation\tf2oda_inference_from_saved_model.py ^
--model_path="exported-models/%MODELNAME%/saved_model/" ^
--image_dir="images/validation" ^
--labelmap="annotations/%LABELMAP%" ^
--detections_out="results/%MODELNAME%/%HARDWARENAME%/detections.csv" ^
--latency_out="results/latency.csv" ^
--min_score=0.5 ^
--model_name=%MODELNAME% ^
--model_short_name=%MODELNAMESHORT% ^
--hardware_name=%HARDWARENAME% ^
--index_save_file="./tmp/index.txt"


