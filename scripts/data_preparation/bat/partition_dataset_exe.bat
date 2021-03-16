echo #====================================#
echo #Prepare dataset for training
echo #====================================#
echo #Partition dataset in a training and validation set

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=C:\Projekte\21_SoC_EML\scripts-and-guides\scripts

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

python %SCRIPTPREFIX%\partition_dataset.py ^
-i "images" ^
--xmlDir "annotations/xmls" ^
--file_id_dir "annotations" ^
-r 0.10 ^
--remove_source

rem #python 020_partition_dataset.py -i "images" --xmlDir "annotations/xmls" --file_id_dir "annotations" -r 0.10