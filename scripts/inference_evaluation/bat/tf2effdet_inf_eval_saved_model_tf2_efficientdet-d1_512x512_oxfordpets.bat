echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USEREMAIL=alexander.wendt@tuwien.ac.at
::set MODELNAME=tf2oda_efficientdetd2_768_576_coco17_pedestrian
set HARDWARENAME=Inteli7Alienware
set PYTHONENV=tf26
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=label_map.pbtxt

set DATASET=../../datasets/oxford_pets_reduced

::Extract the model name from the current file name
set THISFILENAME=%~n0
set MODELNAME=%THISFILENAME:tf2effdet_inf_eval_saved_model_=%
echo Current model name extracted from filename: %MODELNAME%

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo # Infer Images from Known Model
echo #====================================#

echo Inference from model

set PYTHONPATH=C:\Projekte\21_SoC_EML\automl\efficientdet

python %SCRIPTPREFIX%\inference_evaluation\tf2effdet_inference_from_saved_model.py ^
--model_path "exported-models/%MODELNAME%/saved_model/" ^
--image_dir "%DATASET%/images/val" ^
--detections_out="results/%MODELNAME%/%HARDWARENAME%/detections.csv" ^
--latency_out="results/latency.csv" ^
--min_score=0.5 ^
--model_name=%MODELNAME% ^
--hardware_name=%HARDWARENAME% ^
--index_save_file="./tmp/index.txt"

::--model_short_name=%MODELNAMESHORT% unused because the name is created in the csv file

echo #====================================#
echo # Convert Detections to Pascal VOC Format
echo #====================================#
echo Convert TF CSV Format (similar to voc) to Pascal VOC XML

set PYTHONPATH=C:\Projekte\21_SoC_EML\tensorflow_object_detection_api_tf2\models\research;C:\Projekte\21_SoC_EML\tensorflow_object_detection_api_tf2\models\research\slim;C:\Projekte\21_SoC_EML\tensorflow_object_detection_api_tf2\models\research\object_detection

python %SCRIPTPREFIX%\conversion\convert_tfcsv_to_voc.py ^
--annotation_file="results/%MODELNAME%/%HARDWARENAME%/detections.csv" ^
--output_dir="results/%MODELNAME%/%HARDWARENAME%/det_xmls" ^
--labelmap_file="%DATASET%/annotations/%LABELMAP%"


echo #====================================#
echo # Convert to Pycoco Tools JSON Format
echo #====================================#
echo Convert TF CSV to Pycoco Tools csv
python %SCRIPTPREFIX%\conversion\convert_tfcsv_to_pycocodetections.py ^
--annotation_file="results/%MODELNAME%/%HARDWARENAME%/detections.csv" ^
--output_file="results/%MODELNAME%/%HARDWARENAME%/%MODELNAME%_coco_detections.json"

echo #====================================#
echo # Evaluate with Coco Metrics
echo #====================================#

python %SCRIPTPREFIX%\inference_evaluation\objdet_pycoco_evaluation.py ^
--groundtruth_file="%DATASET%/annotations/coco_val_annotations.json" ^
--detection_file="results/%MODELNAME%/%HARDWARENAME%/%MODELNAME%_coco_detections.json" ^
--output_file="results/performance.csv" ^
--model_name=%MODELNAME% ^
--hardware_name=%HARDWARENAME% ^
--index_save_file="./tmp/index.txt"

echo #====================================#
echo # Merge results to one result table
echo #====================================#
echo merge latency and evaluation metrics
python %SCRIPTPREFIX%\inference_evaluation\merge_results.py ^
--latency_file="results/latency.csv" ^
--coco_eval_file="results/performance.csv" ^
--output_file="results/combined_results.csv"

