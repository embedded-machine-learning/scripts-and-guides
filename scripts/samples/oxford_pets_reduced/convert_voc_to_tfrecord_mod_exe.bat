echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\..\

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #======================================================#
echo # Generate TF Records from images and XML as Pascal Voc
echo #======================================================# 

python %SCRIPTPREFIX%\conversion\convert_voc_to_tfrecord_mod.py ^
-x "annotations/xmls" ^
-i "images/train" ^
-l "annotations/pets_label_map.pbtxt" ^
-o "prepared-records/train.record" ^
-n 3

python %SCRIPTPREFIX%\conversion\convert_voc_to_tfrecord_mod.py ^
-x "annotations/xmls" ^
-i "images/validation" ^
-l "annotations/pets_label_map.pbtxt" ^
-o "prepared-records/validation.record" ^
-n 3