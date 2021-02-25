echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_efficientdetd2_768_576_coco17_pedestrian_all
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Convert TF CSV Format (similar to voc) to Pascal VOC XML
echo #======================================================# 

python %SCRIPTPREFIX%\conversion\convert_tfcsv_to_voc.py ^
--annotation_file="results/tf2oda_efficientdetd2_768_576_coco17_pedestrian_all/detections.csv" ^
--output_dir="results/tf2oda_efficientdetd2_768_576_coco17_pedestrian_all/xml" ^
--labelmap_file="annotations/pedestrian_label_map.pbtxt"