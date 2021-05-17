#!/bin/bash

python3 trtexec_build_benchmark.py \
--loadEngine="/media/cdleml/128GB/Users/amozelli/onnx_to_trt/trt_engine/tfkeras_inceptionv3_299x299_imagenet_.engine" \
--shapes='data:1x3x299x299' \
--inputs_dir="/media/cdleml/128GB/Users/amozelli/onnx_to_trt/data/" \
--precision=fp32
