# Training_Scripts
Everything that is needed to train a certain network on a server or locally is provided here. Together with each script, 
you find sample data and a script file (.bat or .sh) that executes on the sample data.

## Pre-requisites
Tensorflow object detection api 2.0 is necessary
Tensorflow 2.4
Requirements file: requirements_tf24odapi.txt

Install TF2 Object Detection API: install_tf2odapi.sh

## Tensorflow 2 Object Detection API
Load TF2 Object Detection API environment: init_eda_env.sh
Load GPU task spooler: init_eda_ts.sh
Lood TF2 Object Detection API environment as well as task spooler: init_eda_ts_env.sh

In the following examples, these constants are used:
```shell
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..
```

### Training Object Detection API
Start training on a CPU or GPU for the Tensorflow 2 Object Detection API.

Source: https://github.com/tensorflow/models/blob/master/research/object_detection/model_main_tf2.py 

Script: `tf2oda_model_main_training.py` 

Example: 
```shell
python %SCRIPTPREFIX%\tf2oda_model_main_training.py ^
--pipeline_config_path="%BASEPATH%/jobs/%MODELNAME%.config" ^
--model_dir="%BASEPATH%/models/%MODELNAME%"
```

### Evaluating Trained Model from Checkpoint
Prints the Coco metrics from the tensorboard prepared file from the training of the last checkpoint

Source: https://github.com/tensorflow/models/blob/master/research/object_detection/

Script: `tf2oda_evaluate_ckpt_performance.py` 

Example: 
```shell
python %SCRIPTPREFIX%\tf2oda_evaluate_ckpt_performance.py ^
--pipeline_config_path="jobs/%modelname%.config" ^
--model_dir="%BASEPATH%/models/%MODELNAME%" ^
--checkpoint_dir="%BASEPATH%/models/%MODELNAME%"
```

### Extract Metrics From Tensorboard to CSV File
Exports the metrics from tensorboard to a CSV file.

Source: https://github.com/tensorflow/models/blob/master/research/object_detection/

Script: `tf2oda_evaluate_ckpt_performance.py` 

Example: 
```shell
python %SCRIPTPREFIX%\tf2oda_read_tf_summary.py ^
--checkpoint_dir="%BASEPATH%/models/%MODELNAME%" ^
--out_dir="result/%MODELNAME%/metrics"
```

### Export a TF2 Saved Model from a Checkpoint
Export a TF2 Saved Model from a Checkpoint. The latest checkpoint of a folder is selected. If you want another checkpoint, edit the file "checkpoint" and 
enter the checkpoint to export in the line ```model_checkpoint_path: "ckpt-5"```

Source: https://github.com/tensorflow/models/blob/master/research/object_detection/

Script: `tf2oda_export_savedmodel.py` 

Example: 
```shell
python %SCRIPTPREFIX%\tf2oda_export_savedmodel.py ^
--input_type="image_tensor" ^
--pipeline_config_path="%BASEPATH%/jobs/%MODELNAME%.config" ^
--trained_checkpoint_dir="%BASEPATH%/models/%MODELNAME%" ^
--output_directory="exported-models/%MODELNAME%"
```

### Zip folders and files into a Zip file
Zip files and folders into a zip file. Wildcards can be used to select all files of a certain type. Within this folder, scripts for automatic 
compressing of a workspace are included. The compressed file can be copied to the training server and processed from there.

Source: 

Script: `zip_tool.py` 

Example: 
```shell
python %SCRIPTPREFIX%\zip_tool.py ^
--items="jobs, *.sh" ^
--out="tmp/jobs.zip"
```

# Issues
If there are any issues or suggestions for improvements, please add an issue to github's bug tracking system or please send a mail 
to [Alexander Wendt](mailto:alexander.wendt@tuwien.ac.at)