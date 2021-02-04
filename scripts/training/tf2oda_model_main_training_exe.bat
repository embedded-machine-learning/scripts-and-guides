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
echo # Model evaluation and export
echo #====================================#

echo #====================================#
echo #Train model
echo #====================================#
echo model used %MODELNAME%

python %SCRIPTPREFIX%\tf2oda_model_main_training.py --pipeline_config_path="%BASEPATH%/jobs/%MODELNAME%.config" --model_dir="%BASEPATH%/models/%MODELNAME%"
:: echo ## Evaluate during training (in another window or in background)
:: python 030_model_main_tf2.py --pipeline_config_path="jobs/%modelname%.config" --model_dir="models/%modelname%" --checkpoint_dir="models/%modelname%"

echo #====================================#
echo #Evaluate trained model
echo #====================================#
:: python 040_evaluate_final_performance.py --pipeline_config_path="jobs/%modelname%.config" --model_dir="models/%modelname%" --checkpoint_dir="models/%modelname%"
::python 041_read_tf_summary.py --checkpoint_dir="models/%modelname%" --out_dir="result/metrics"

echo #====================================#
echo #Export inference graph
echo #====================================#
:: python 050_exporter_main_v2.py --input_type="image_tensor" --pipeline_config_path="jobs/%modelname%.config" --trained_checkpoint_dir="models/%modelname%" --output_directory="exported-models/%modelname%"




::echo "Training of %MODELNAME% finished" | mail -s "Training of %MODELNAME% finished" %USEREMAIL%

echo #======================================================#
echo # Training, evaluation and export of the model completed
echo #======================================================# 