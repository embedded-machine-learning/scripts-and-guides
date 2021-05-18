#!/bin/bash

python3 trtexec_build_benchmark.py \
--onnx="/media/cdleml/128GB/Users/amozelli/onnx_to_trt/onnx_models/tfkeras_inceptionv3_299x299_imagenet_.onnx" \
--precision=fp32 \
--buildEngine True


