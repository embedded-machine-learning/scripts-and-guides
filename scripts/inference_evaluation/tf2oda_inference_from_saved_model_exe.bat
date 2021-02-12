echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constants Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=.
set LABELMAPNAME=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo #Infer new images
echo #====================================#

python %SCRIPTPREFIX%\tf2oda_inference_from_saved_model.py ^
--model_path "../training/samples/oxford_pets_reduced/exported-models/%MODELNAME%/saved_model/" ^
--image_dir "../training/samples/oxford_pets_reduced/images/validation" ^
--labelmap "../training/samples/oxford_pets_reduced/annotations/%LABELMAPNAME%" ^
--output_dir="../training/samples/oxford_pets_reduced/results/%MODELNAME%" ^
--run_detection True 

