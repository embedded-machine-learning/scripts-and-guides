echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo # clean dataset
echo #====================================#

echo Clean images from images with wrong formats, which causes error in the training phase

python %SCRIPTPREFIX%\data_preparation\clean_images.py ^
-i "images"