echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constants Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
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
--groundtruth_path="annotations/coco_pets_validation_annotations.json"
--detection_path="results/%MODELNAME%/validation_for_inference/coco_pets_detection_annotations.json
::--labelmap="samples/annotations/label_map.pbtxt" ^
::--output_dir="samples/results" ^
::--image_path1="samples/images/0.jpg" --annotation_dir1="samples/annotations/xml" --title1="Image 1" ^
::--image_path2="samples/images/10.jpg" --annotation_dir2="samples/annotations/xml" --title2="Image 2" ^
::--image_path3="samples/images/20.jpg" --annotation_dir3="samples/annotations/xml" --title3="Image 3" ^
::--use_three_images
