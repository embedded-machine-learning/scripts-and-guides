echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

echo INFO: EXECUTE SCRIPT IN TARGET BASE FOLDER, e.g. samples/oxford_pets_reduced

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_pets
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
echo Evaluate checkpoint performance by testing the training set with the validation set
python %SCRIPTPREFIX%\tf2oda_evaluate_ckpt_performance.py ^
--pipeline_config_path="jobs/%modelname%.config" ^
--model_dir="%BASEPATH%/models/%MODELNAME%" ^
--checkpoint_dir="%BASEPATH%/models/%MODELNAME%"

echo Read TF Summary of the validation from Tensorboard file

python %SCRIPTPREFIX%\tf2oda_read_tf_summary.py ^
--checkpoint_dir="%BASEPATH%/models/%MODELNAME%" ^
--out_dir="result/%MODELNAME%/metrics"

echo #====================================#
echo #Export inference graph
echo #====================================#
echo Export model %modelname%
python %SCRIPTPREFIX%\tf2oda_export_savedmodel.py ^
--input_type="image_tensor" ^
--pipeline_config_path="%BASEPATH%/jobs/%MODELNAME%.config" ^
--trained_checkpoint_dir="%BASEPATH%/models/%MODELNAME%" ^
--output_directory="exported-models/%MODELNAME%"

::echo "Stop Training of %MODELNAME%" | mail -s "Training of %MODELNAME% finished" %USEREMAIL%

echo #======================================================#
echo # Training, evaluation and export of the model completed
echo #======================================================# 