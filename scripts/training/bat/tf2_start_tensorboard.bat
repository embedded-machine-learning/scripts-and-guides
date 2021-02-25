echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/starwars_reduced
echo INFO: Start tensorboard in a separate shell

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

::echo "Activate python environment and add python path variables" $PYTHONENV
::source $PYTHONENV

::echo "Start training of %MODELNAME%" | mail -s "Start training of %MODELNAME%" %USEREMAIL%

echo #======================================================#
echo # Open Tensorboard for Training Data
echo #======================================================# 
start cmd /c "tensorboard --logdir models/%MODELNAME%/train --port=6006 --bind_all" 

timeout 20

echo open browser with tensorboard on address http://localhost:6006
start http://localhost:6006

echo close tensorbaord with Ctrl+C

echo #======================================================#
echo # Tensorboard close
echo #======================================================# 