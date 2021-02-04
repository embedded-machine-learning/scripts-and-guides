echo #===========================================#
echo # Embedded Machine Learning Toolbox         #
echo #===========================================#

rem Set Varibles
set modelname=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
rem set config_file="config/omxs30_lt.ini"
set script_root="..\.."
set env="tf24"

echo setup environment %env%
call conda activate %env%

echo #===========================================#
echo # Train dataset                             #
echo #===========================================#

python %script_root%\tf2_model_main_training.py --pipeline_config_path="jobs/%modelname%.config" --model_dir="models/%modelname%"

