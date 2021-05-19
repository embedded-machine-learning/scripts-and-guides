echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constants Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
set PYTHONENV=tf24
set BASEPATH=..\samples\inference_evaluation
::set BASEPATH=.\
set SCRIPTPREFIX=..\..
::set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pets_label_map.pbtxt

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

:: Basepath preparation
cd %BASEPATH%

echo #====================================#
echo # Visualize Networks
echo #====================================#

python %SCRIPTPREFIX%\inference_evaluation\evaluation_visualization.py ^
--input_combined_file=result_best_hardware.csv ^
--output_dir="./images" ^
--hwopt_reference="" ^
--latency_requirement=100

:: Alternative for 2 different files. two different files will be removed in later versions.
::python %SCRIPTPREFIX%\inference_evaluation\evaluation_visualization.py ^
::--latency_file="results/latency.csv" ^
::--performance_file="results/performance.csv" ^
::--output_dir="results" ^
::--hwopt_reference="" ^
::--latency_requirement=100
