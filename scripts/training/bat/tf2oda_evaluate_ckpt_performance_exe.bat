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
echo #Evaluate trained model
echo #====================================#
echo Evaluate checkpoint performance
python %SCRIPTPREFIX%\tf2oda_evaluate_ckpt_performance.py ^
--pipeline_config_path="jobs/%modelname%.config" ^
--model_dir="%BASEPATH%/models/%MODELNAME%" ^
--checkpoint_dir="%BASEPATH%/models/%MODELNAME%"

::python tf2oda_read_tf_summary.py --checkpoint_dir="models/%modelname%" --out_dir="result/metrics"


::echo "Training of %MODELNAME% finished" | mail -s "Training of %MODELNAME% finished" %USEREMAIL%

echo #======================================================#
echo #Training, evaluation and export of the model completed
echo #======================================================# 