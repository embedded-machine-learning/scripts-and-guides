echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constants Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo #Infer new images
echo #====================================#

python %SCRIPTPREFIX%\inference_evaluation\tf2oda_inference_from_saved_model.py ^
--model_path "exported-models/%MODELNAME%/saved_model/" ^
--image_dir "images/validation" ^
--labelmap "annotations/%LABELMAP%" ^
--output_dir="results/%MODELNAME%/validation_for_inference" ^
--xml_dir="results/%MODELNAME%/validation_for_inference" ^
--run_detection True ^
--run_visualization True ^
--min_score=0.5 ^
--model_name=%MODELNAME% ^
--hardware_name="CPU_Intel_i5"

