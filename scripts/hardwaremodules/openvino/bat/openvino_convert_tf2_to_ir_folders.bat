@echo off

:main
:: Constant Definition
set USEREMAIL=alexander.wendt@tuwien.ac.at
::set MODELNAME=tf2oda_ssdmobilenetv2_320x320_pedestrian
::set MODELNAME=tf2oda_ssdmobilenetv2_300x300_pets
::set MODELNAME=tf2oda_efficientdet_512x512_pedestrian_D0_LR02
set PYTHONENV=tf24
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pets_label_map.pbtxt

:: OpenVino Input
set OPENVINOINSTALLDIR="C:\Projekte\21_SoC_EML\openvino"
set SETUPVARS="C:\Projekte\21_SoC_EML\openvino\scripts\setupvars\setupvars.bat"

::======== Get model from file name ===============================
::Extract the model name from the current file name
::set THISFILENAME=%~n0
::set MODELNAME=%THISFILENAME:openvino_convert_tf2_to_ir_=%
::echo Current model name extracted from filename: %MODELNAME%
::======== Get file name ===============================

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

:: Setup OpenVino Variables
echo Setup OpenVino Variables %SETUPVARS%
call %SETUPVARS%

::Use this method to use all folder names in the subfolder as models
echo Convert files in the folder exported-models

set MODELFOLDER=exported-models
for /d %%D in (%MODELFOLDER%\*) do (
	::For each folder name in exported models, 
	set MODELNAME=%%~nxD
	for %%x in (FP16 FP32) do (
		::For each possible quantization
		set PRECISION=%%x
		call :perform_conversion
	)
)
goto :eof

::call :perform_conversion
::===================================================================::

:::test
::echo XApply to model %MODELNAME% with precision %PRECISION%
::goto :eof

:perform_conversion

echo Apply to model %MODELNAME% with precision %PRECISION%

echo #====================================#
echo # Convert TF2 Model to OpenVino Intermediate Representation
echo #====================================#
echo "Define API config file"
set APIFILEEFF=%OPENVINOINSTALLDIR%\model-optimizer\extensions\front\tf\efficient_det_support_api_v2.4.json
set APIFILESSD=%OPENVINOINSTALLDIR%\model-optimizer\extensions\front\tf\\ssd_support_api_v2.4.json
set APIFILE=ERROR

if not x%MODELNAME:ssd=%==x%MODELNAME% (set APIFILE=%APIFILESSD%)
if not x%MODELNAME:effi=%==x%MODELNAME% (set APIFILE=%APIFILEEFF%)
echo Use this API file: %APIFILE%

echo "Start conversion"
python %OPENVINOINSTALLDIR%\model-optimizer\mo_tf.py ^
--saved_model_dir="exported-models\%MODELNAME%\saved_model" ^
--tensorflow_object_detection_api_pipeline_config=exported-models\%MODELNAME%\pipeline.config ^
--transformations_config=%APIFILE% ^
--reverse_input_channels ^
--data_type %PRECISION% ^
--output_dir=exported-models-openvino\%MODELNAME%_OV%PRECISION%
::--tensorflow_use_custom_operations_config=openvinofiles\%APIFILE% ^

::--data_type {FP16,FP32,half,float}
::                         Data type for all intermediate tensors and weights. If
::                         original model is in FP32 and --data_type=FP16 is
::                         specified, all model weights and biases are quantized
::                         to FP16.


::OLD
::set MODELDIR=./pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model
::set PIPELINEDIR=./pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config
::set PYTHONPATH=/home/$USERNAME/environments/$ENVIRONMENTNAME

::exported-models\tf2oda_ssdmobilenetv2_320x320_pedestrian\saved_model
::exported-models\tf2oda_ssdmobilenetv2_320x320_pedestrian\pipeline.config

::call %OPENVINOINSTALLDIR%\scripts\setupvars\setupvars.bat
::call %OPENVINOINSTALLDIR%\bin\setupvars.bat

::set SETUPVARS="C:\Program Files (x86)\Intel\openvino_2021.3.394\bin\setupvars.bat"
::set OPENVINOINSTALLDIR="C:\Program Files (x86)\Intel\openvino_2021.3.394"

::python %OPENVINOINSTALLDIR%\deployment_tools\model_optimizer\mo_tf.py ^
::python %OPENVINOINSTALLDIR%\model-optimizer\mo_tf.py ^
::--saved_model_dir=%MODELDIR% ^
::--tensorflow_object_detection_api_pipeline_config=%PIPELINEDIR% ^
::--reverse_input_channels ^
::--tensorflow_use_custom_operations_config=openvinofiles\ssd_support_api_v2.4.json
::--input_shape=[1,320,320,3] ^
::--log_level=DEBUG
::--transformations_config=openvinofiles\ssd_support_api_v2.4.json ^

::--tensorflow_use_custom_operations_config=ssd_support_api_v2.0.json ^

::python %OPENVINOINSTALLDIR%\model-optimizer\mo_tf.py ^
::--saved_model_dir %MODELDIR% ^
::--tensorflow_object_detection_api_pipeline_config %PIPELINEDIR% ^
::--tensorflow_use_custom_operations_config ssd_support_api_v2.0.json

:: Working Pipeline
::python C:\Projekte\21_SoC_EML\openvino\model-optimizer\mo_tf.py --saved_model_dir="exported-models\tf2oda_ssdmobilenetv2_320x320_pedestrian\saved_model" --tensorflow_use_custom_operations_config=openvinofiles\ssd_support_api_v2.4.json --tensorflow_object_detection_api_pipeline_config=exported-models\tf2oda_ssdmobilenetv2_320x320_pedestrian\pipeline.config --reverse_input_channels

echo "Conversion finished"
goto :eof

:end