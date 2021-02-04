# Training_Scripts
Everything that is needed to train a certain network on a server or locally is provided here. Together with each script, 
you find sample data and a script file (.bat or .sh) that executes on the sample data.

## Pre-requisites
Tensorflow object detection api 2.0 is necessary
Tensorflow 2.4

## Tensorflow 2 Object Detection API
Start training on a CPU or GPU for the Tensorflow 2 Object Detection API.

Source: https://github.com/tensorflow/models/blob/master/research/object_detection/model_main_tf2.py 

Script: `tf2oda_model_main_training.py` 

Example: 
```shell
python %SCRIPTPREFIX%\tf2oda_model_main_training.py ^
--pipeline_config_path="%BASEPATH%/jobs/%MODELNAME%.config" ^
--model_dir="%BASEPATH%/models/%MODELNAME%"
```

# Issues
If there are any issues or suggestions for improvements, please add an issue to github's bug tracking system or please send a mail 
to [Alexander Wendt](mailto:alexander.wendt@tuwien.ac.at)