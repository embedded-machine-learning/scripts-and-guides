echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

::echo "Activate python environment and add python path variables" $PYTHONENV
::source $PYTHONENV

::echo "Start training of %MODELNAME%" | mail -s "Start training of %MODELNAME%" %USEREMAIL%


echo #====================================#
echo #Train model
echo #====================================#
echo model used %MODELNAME%

python %SCRIPTPREFIX%\tf2oda_model_main_training.py ^
--pipeline_config_path="%BASEPATH%/jobs/%MODELNAME%.config" ^
--model_dir="%BASEPATH%/models/%MODELNAME%"

::echo "Training of %MODELNAME% finished" | mail -s "Training of %MODELNAME% finished" %USEREMAIL%

echo #======================================================#
echo # Training, evaluation and export of the model completed
echo #======================================================# 