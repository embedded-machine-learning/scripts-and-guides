#!/bin/sh

PRECISION = INT8

python3 convert_and_benchmark_trt.py \
--tensorflow_model=./tf2oda_ssdmobilenetv2_320x320_coco17_D100_pedestrian_test/saved_model/ \
--precision='INT8' \
--image_size='[320,320]' \
--batch_size=1 \
--data_dir='./data/' \
--net=./tf2oda_ssdmobilenetv2_320x320_coco17_D100_pedestrian_test/saved_model/_TFTRT_FP32 \
--csv=./results.csv
#--conversion


#python3 convert_and_benchmark_trt.py \
#--tensorflow_model='./tf2oda_ssdmobilenetv2_320x320_coco17_D100_pedestrian_test/saved_model' \
#--net='./test' \
#--batch_size=1 \
#--image_size='[320,320]' \
#--precision='FP32' \
#--conversion \
#--data_dir='./data'