echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\Projekte\21_SoC_EML\scripts-and-guides\scripts

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Generate TF Records from images and XML as Pascal Voc
echo #======================================================# 

python %SCRIPTPREFIX%\conversion\convert_voc_to_tfrecord_mod.py ^
-x "samples/annotations/xml" ^
-i "samples/images" ^
-l "samples/annotations/sw_label_map.pbtxt" ^
-o "samples/prepared-records/train_voc.record" ^
-n 2