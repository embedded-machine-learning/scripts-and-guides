::# Example constant definitions please update

:: Constant Definition
set USEREMAIL=alexander.wendt@tuwien.ac.at
::set MODELNAME=tf2oda_ssdmobilenetv2_320x320_pedestrian
set MODELNAME=tf2oda_ssdmobilenetv2_300x300_pets
::set MODELNAME=tf2oda_efficientdet_512x512_pedestrian_D0_LR02
set PYTHONENV=tf24
set SCRIPTPREFIX=..\..\scripts-and-guides\scripts
set LABELMAP=pets_label_map.pbtxt
set HARDWARENAME=Inteli7dp3510

::OpenVino Constant Defintion
::Inference uses a different version than model conversion.
set OPENVINOINSTALLDIR="C:\Program Files (x86)\Intel\openvino_2021.2.185"
set SETUPVARS="C:\Program Files (x86)\Intel\openvino_2021.2.185\bin\setupvars.bat"
::set OPENVINOINSTALLDIR="C:\Projekte\21_SoC_EML\openvino"
::set SETUPVARS="C:\Projekte\21_SoC_EML\openvino\scripts\setupvars\setupvars.bat"

set MODELDIR=exported-models\%MODELNAME%\saved_model
set PIPELINEDIR=exported-models\%MODELNAME%\pipeline.config
::set APIFILE=..\..\scripts-and-guides\scripts\hardwaremodules\openvino\openvino_conversion_config\efficient_det_support_api_v2.4.json
set APIFILE=..\..\scripts-and-guides\scripts\hardwaremodules\openvino\openvino_conversion_config\ssd_support_api_v2.4.json


::Extract the model name from the current file name
set THISFILENAME=%~n0
set MODELNAME=%THISFILENAME:openvino_infer_latency_=%
echo Current model name extracted from filename: %MODELNAME%

::powershell: cmd /c '"C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"'
:: cmd: "C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

:: Setup OpenVino Variables
echo Setup OpenVino Variables %SETUPVARS%
call %SETUPVARS%


echo #====================================#
echo # Infer with OpenVino
echo #====================================#
echo "Start latency inference"
python %SCRIPTPREFIX%\hardwaremodules\openvino\run_pb_bench_sizes.py ^
-openvino_path %OPENVINOINSTALLDIR% ^
-hw CPU ^
-batch_size 1 ^
-api sync ^
-niter 100 ^
-xml exported-models-openvino/%MODELNAME%/saved_model.xml ^
-output_dir="results/%MODELNAME%/%HARDWARENAME%/OpenVino"

::-size [1,320,320,3] ^


::-hw (CPU|MYRIAD)
::-size (batch, width, height, channels=3)
::-pb Frozen file

echo #====================================#
echo # Convert Latencies
echo #====================================#
echo "Add measured latencies to result table"
python %SCRIPTPREFIX%\hardwaremodules\openvino\latency_parser\openvino_latency_parser.py ^
--avg_rep results/%MODELNAME%/%HARDWARENAME%/OpenVino_sync\benchmark_average_counters_report_saved_model_CPU_sync.csv ^
--inf_rep results/%MODELNAME%/%HARDWARENAME%/OpenVino_sync\benchmark_report_saved_model_CPU_sync.csv ^
--output_path results/latency.csv

::--save_new #Always append

echo "Inference finished"