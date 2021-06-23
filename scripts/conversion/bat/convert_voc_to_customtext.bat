@echo off

:main

echo #==============================================#
echo # CDLEML Process
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

echo #====================================#
echo # Convert COV to Custom Text format that is used in https://github.com/david8862/keras-YOLOv3-model-set
echo #====================================#

for %%x in (train validation) do (
		::For each possible quantization
		set TYPE=%%x
		call :convert
	)
	
goto :eof

:convert
echo Convert %TYPE% data
python %SCRIPTPREFIX%\conversion\convert_voc_to_customtext.py ^
--annotations_dir=annotations/xmls ^
--image_dir=images/%TYPE% ^
--output_path=annotations/yolo/yolo_%TYPE%.txt ^
--classes_path=annotations/labels.txt ^
--include_difficult ^
--include_no_obj

goto :eof

:end