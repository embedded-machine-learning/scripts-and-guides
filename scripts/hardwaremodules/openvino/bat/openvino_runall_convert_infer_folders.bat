echo Execute conversion and inference

call openvino_convert_tf2_to_ir_folders.bat
call openvino_infer_latency_folders.bat