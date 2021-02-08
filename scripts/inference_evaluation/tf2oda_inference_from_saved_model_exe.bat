echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constants Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=.

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo #Infer new images
echo #====================================#

python %SCRIPTPREFIX%\tf2oda_inference_from_saved_model.py ^
--model_path "../training/samples/starwars_reduced/exported-models/%MODELNAME%/saved_model/" ^
--image_dir "../training/samples/starwars_reduced/images/test" ^
--labelmap "../training/samples/starwars_reduced/annotations/sw_label_map.pbtxt" ^
--output_dir="../training/samples/starwars_reduced/result/%MODELNAME%" ^
--run_detection True 

