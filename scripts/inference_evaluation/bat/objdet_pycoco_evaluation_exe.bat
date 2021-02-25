echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constants Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set MODELNAMESHORT=mobnetv2300_D100
set HARDWARE=Intel_CPU_i5
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\..
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo # Evaluate with Coco Metrics
echo #====================================#

python %SCRIPTPREFIX%\inference_evaluation\objdet_pycoco_evaluation.py ^
--groundtruth_file="annotations/coco_pets_validation_annotations.json" ^
--detection_file="results/%MODELNAME%/validation_for_inference/coco_pets_detection_annotations.json" ^
--output_file="results/performance.csv" ^
--model_name=%MODELNAME% ^
--model_short_name=%MODELNAMESHORT% ^
--hardware_name=%HARDWARE%
