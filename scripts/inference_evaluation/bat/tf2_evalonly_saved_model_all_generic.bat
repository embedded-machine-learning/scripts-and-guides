@echo off

:main

@echo #==============================================#
@echo # CDLEML Process TF2 Object Detection API
@echo #==============================================#

:: Constant Definition
set USEREMAIL=alexander.wendt@tuwien.ac.at
::set MODELNAME=tf2oda_efficientdetd2_768_576_coco17_pedestrian
set HARDWARENAME=Inteli7dp3510
set PYTHONENV=tf24
::set SCRIPTPREFIX=..\..\..
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pedestrian_label_map.pbtxt


::Extract the model name from the current file name
::set THISFILENAME=%~n0
::set MODELNAME=%THISFILENAME:tf2oda_inference_and_evaluation_from_saved_model_=%
::echo Current model name extracted from filename: %MODELNAME%

:: Environment preparation
@echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

:: only name %%~nxD
:: full path %%~fD
:: THISFILENAME=%~n0

::Use this method to use all folder names in the subfolder as models
set MODELFOLDER=exported-models\_unfinished
for /d %%D in (%MODELFOLDER%\*) do (
	::For each folder name in exported models, 
	set MODELNAME=%%~nxD
	call :perform_inference
)

:: Use this methods to iterate through a list MODELS
::SET MODELS=^
::MODEL1 ^
::MODELZ

::for %%x in (%MODELS%) do (
::		set MODELNAME=%%x
::		call :perform_inference
::       )


goto :end





:perform_inference
echo Apply to model %MODELNAME%

::echo #====================================#
::echo # Infer Images from Known Model
::echo #====================================#

::echo Inference from model 
::python %SCRIPTPREFIX%\inference_evaluation\tf2oda_inference_from_saved_model.py ^
::--model_path "exported-models/%MODELNAME%/saved_model/" ^
::--image_dir "images/validation" ^
::--labelmap "annotations/%LABELMAP%" ^
::--detections_out="results/%MODELNAME%/%HARDWARENAME%/detections.csv" ^
::--latency_out="results/latency_%HARDWARENAME%.csv" ^
::--min_score=0.5 ^
::--model_name=%MODELNAME% ^
::--hardware_name=%HARDWARENAME%

::--model_short_name=%MODELNAMESHORT% unused because the name is created in the csv file


echo #====================================#
echo # Convert Detections to Pascal VOC Format
echo #====================================#
echo Convert TF CSV Format (similar to voc) to Pascal VOC XML
python %SCRIPTPREFIX%\conversion\convert_tfcsv_to_voc.py ^
--annotation_file="results/%MODELNAME%/%HARDWARENAME%/detections.csv" ^
--output_dir="results/%MODELNAME%/%HARDWARENAME%/det_xmls" ^
--labelmap_file="annotations/%LABELMAP%"


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
--groundtruth_file="annotations/coco_pets_validation_annotations.json" ^
--detection_file="results/%MODELNAME%/%HARDWARENAME%/%MODELNAME%_coco_detections.json" ^
--output_file="results/performance_%HARDWARENAME%.csv" ^
--model_name=%MODELNAME% ^
--hardware_name=%HARDWARENAME%

echo #====================================#
echo # Move executed models to exported inferred
echo #====================================#
md exported-models-inferred
call move .\exported-models\%MODELNAME% exported-models-inferred

goto :eof

:end