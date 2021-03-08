#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert and infer into Tensor-rt models

License_info:
# ==============================================================================
# ISC License (ISC)
# Copyright 2020 Christian Doppler Laboratory for Embedded Machine Learning
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ==============================================================================

# The following script uses several method fragments the following guide
# Source: https://towardsai.net/p/deep-learning/cvml-annotation%e2%80%8a-%e2%80%8awhat-it-is-and-how-to-convert-it

"""

# Futures
#from __future__ import print_function
#from __future__ import absolute_import, division, print_function, unicode_literals

# Built-in/Generic Imports
import csv
import datetime
import json
import os
import time
import argparse

# Libs
import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# Own modules

__author__ = 'Amid Mozelli'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Alexander Wendt', 'Rohit Verma']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Amid Mozelli'
__email__ = 'amid.mozelli@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Benchmarking TRT-Optimized TF-Models')

parser.add_argument('-n', '--net', default='./inceptionv3_saved_model_TFTRT_FP16', help='Saved optimized model path',
                    required=False)
					
parser.add_argument('-t', '--tensorflow_model', default='./inceptionv3_saved_model',
                    help='Unoptimized Tensorflow model', required=False)
					
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Batch Size', required=False)
					
parser.add_argument('-s', '--image_size', type=str, default='[299, 299]',
                    help='List of two coordinates: [Height, Width]', required=False)
					
parser.add_argument('-p', '--precision', default='FP32',
                    help='TensorRT precision mode: FP32, FP16 or INT8.', required=False)

parser.add_argument('-e', '--dtype', default='float32',
                    help='Data type for the input from float32, float16 or uint8.', required=False)
					
parser.add_argument('-i', '--image_predict', default='./data/img0.JPG',
                    help='Single image prediction.', required=False)
					
parser.add_argument('-c', '--conversion', action='store_true',
                    help='Set this flag if you want to convert your model from TF to TRT.', required=False)
					
parser.add_argument('-d', '--data_dir', default='./data/',
                    help='Location of the dataset.', required=False)
					
parser.add_argument('-x', '--csv', default='./results.csv',
                    help='Location of the .csv file if exists.', required=False)
					
parser.add_argument('-v', '--visualize', action='store_true',
                    help='Set this flag if you want to visualize one of the predictions.', required=False)
					
args, unknown = parser.parse_known_args()
print(args)


def batch_input(batch_size, data_path, d_type, hw, is_keras=False):
    if d_type == 'float32':
        datatype = np.float32
    elif d_type == 'float16':
        datatype = np.float16
    elif d_type == 'uint8':
        datatype = np.uint8
    else:
        raise ValueError("No valid data type provided: " + d_type + ". It has to be float32, float16 or uint8")


    batched_input = np.zeros((batch_size, hw[0], hw[1], 3), dtype=datatype)
    pics = os.listdir(data_path)
    n = len(pics)

    for i in range(batch_size):
        img_path = data_path + pics[i % n] #generating batches
        img = image.load_img(img_path, target_size=(hw[0], hw[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if is_keras:
            x = preprocess_input(x) #for models loaded from Keras applications
        batched_input[i, :] = x

    batched_input = tf.constant(batched_input)
    return batched_input


def load_tf_saved_model(input_saved_model_dir):
    print(f'Loading saved model {input_saved_model_dir}...')
    start_time = time.time()
    saved_model_loaded = tf.saved_model.load(input_saved_model_dir, tags=[tag_constants.SERVING])
    end_time = time.time()
    print('Loading model took {:4.1f}s'.format(end_time - start_time))
    return saved_model_loaded



#def predict_and_benchmark_throughput(batched_input, infer, args, N_warmup_run=50, N_run=1000):
def predict_and_benchmark_throughput(batched_input, infer, args, N_warmup_run=50, N_run=1000):
    elapsed_time = []
    all_preds = []
    boxes = []
    classes = []
    scores = []
    batch_size = batched_input.shape[0]
    
    print("Running warm up runs...")
    for i in range(N_warmup_run):
        labeling = infer(batched_input)
        #print("Inference {}/{}".format(i, N_warmup_run))
        #preds = labeling['predictions'].numpy() 
        preds = labeling

    print("Running real runs...")
    for i in range(N_run):
        start_time = time.time()

        labeling = infer(batched_input)
        #preds = labeling['predictions'].numpy() 
        preds = labeling

        end_time = time.time()

        elapsed_time = np.append(elapsed_time, end_time - start_time)

        all_preds.append(preds)

        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i + 50, (elapsed_time[-50:].mean()) * 1000))
	
    throughput = N_run * batch_size / elapsed_time.sum()

    #decoding predictions
    data_path = args.data_dir
    pics = os.listdir(data_path)
    hw = json.loads(args.image_size)
    detection_boxes = []
    detection_scores = []
    detection_classes = []
    score = []
    for pic in pics:
        img_path = data_path + pic
        img = image.load_img(img_path, target_size=(hw[0], hw[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(args.dtype)
        x = tf.constant(x)
        pred = infer(x)
        detection_boxes.append(pred['detection_boxes'].numpy()[0])
        for x in pred['detection_scores'].numpy()[0]:
            if x > 0.02:
                score.append(x)
        detection_scores.append(np.array(score))
        #detection_scores.append(pred['detection_scores'].numpy()[0])
        detection_classes.append(pred['detection_classes'].numpy()[0])


    # writing csv file
    now = datetime.datetime.now()
    name_list = args.net.split('_')
    while len(name_list) < 4:
        name_list.append('ND')

    headers =     ['Date',
                    'Model',
                    'Model_Short',
                    'Framework',
                    'Network',
                    'Resolution',
                    'Dataset',
                    'Hardware',
                    'Hardware_Optimization',
                    'Precision',
                    'Batch_Size',
                    'Throughput',
                    'Mean_Latency',
                    'Latencies',
                    'detection_boxes',
                    'detection_scores',
                    'detection_classes']

    body =        [now.strftime('%Y-%m-%d'),# %H:%M:%S"),
                    "MODEL_TBD",
                    "MODEL_SHORT_TBD",
                    name_list[0],
                    name_list[1],
                    name_list[2],
                    name_list[3],
                    "NVIDIA_XAVIER_TBD",
                    "TRT_TBD",
                    args.precision,
                    args.batch_size,
                    '{:.0f} images/s'.format(throughput),
                    '{:.1f} ms'.format(elapsed_time.mean() * 1000),
                    "SINGLE_LATENCIES_TBD",
                    detection_boxes,
                    detection_scores,
                    detection_classes]
        
    if os.path.exists(args.csv):
        with open(args.csv, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(body)
                      
    else:
        with open(args.csv, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerow(body)            


    print('Throughput: {:.0f} images/s'.format(throughput))
    return all_preds
#showing the prediction for the whole dataset
# def show_predictions(model, HW=[299,299], data_path = './data/'):
#     pics = os.listdir(data_path)
#     for i,pic in enumerate(pics):
#         img_path = data_path + pic
#         img = image.load_img(img_path, target_size=(HW[0], HW[1]))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#
#         preds = model.predict(x)
#         print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
#
#         plt.subplot(2,2,i+1)
#         plt.imshow(img);
#         plt.axis('off');
#         plt.title(decode_predictions(preds, top=3)[0][0][1])

# #single image prediction
# def show_prediction(model, HW, img_path):
#     img = image.load_img(img_path, target_size=(HW[0], HW[1]))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     x = tf.constant(x)
#
#     labeling = model(x)
#     preds = labeling['predictions'].numpy()
#
#     # decode the results into a list of tuples (class, description, probability)
#     # (one such list for each sample in the batch)
#     print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
#     plt.subplot(2, 2, 1)
#     plt.imshow(img);
#     plt.axis('off');
#     plt.title(decode_predictions(preds, top=3)[0][0][1])



def convert_to_trt_graph_and_save(precision_mode, input_saved_model_dir, calibration_data):
    if precision_mode == 'FP32':
        precision_mode = trt.TrtPrecisionMode.FP32
        #converted_saved__suffix = '_TFTRT_FP32'
        converted_saved__prefix = 'TRTFP32_'

    if precision_mode == 'FP16':
        precision_mode = trt.TrtPrecisionMode.FP16
        #converted_saved__suffix = '_TFTRT_FP16'
        converted_saved__prefix = 'TRTFP16_'

    if precision_mode == 'INT8':
        precision_mode = trt.TrtPrecisionMode.INT8
        #converted_saved__suffix ='_TFTRT_INT8'
        converted_saved__prefix = 'TRTINT8_'

    #output_saved_model_dir = input_saved_model_dir + converted_saved__suffix
    r = input_saved_model_dir.split('_')
    final = input_saved_model_dir[len(r[0])+1:]
    output_saved_model_dir = converted_saved__prefix + final 

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=8000000000
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params
    )

    print(f'Converting {input_saved_model_dir} to TF-TRT graph precision mode {precision_mode}...')

    if precision_mode == trt.TrtPrecisionMode.INT8:
        def calibration_input_fn():
            yield (calibration_data, )
        start_time = time.time()
        converter.convert(calibration_input_fn=calibration_input_fn)
        end_time = time.time()
    else:
        start_time = time.time()
        converter.convert()
        end_time = time.time()

    print('Conversion took {:4.1f}s.'.format(end_time - start_time))
    print(f'Saving converted model to {output_saved_model_dir}')
    converter.save(output_saved_model_dir=output_saved_model_dir)
    print('Complete')




def main():

    #Image size
    images_size = json.loads(args.image_size)

    #making batched inputs
    print("=== Prepare batch input ===")
    batched_input = batch_input(args.batch_size, args.data_dir, args.dtype, images_size, is_keras=False)
    print("=== Batch input prepared ===")

	
	#conversion
    if (args.conversion):
        print("=== Convert input model to trt. Model={}, Precision={} ===".format(args.tensorflow_model, args.precision))
        convert_to_trt_graph_and_save(args.precision, args.tensorflow_model, batched_input)
        print("=== Conversion complete ===")
    #performing the benchmarking
    else:
        print("=== Load saved model ===")
        saved_model_loaded = load_tf_saved_model(args.net)
        print("=== Model loaded ===")
        infer = saved_model_loaded.signatures['serving_default']
        all_preds = predict_and_benchmark_throughput(batched_input, infer, args, N_warmup_run=50, N_run=1000)
        print("=== Inference complete ===")
    if (args.visualize):
        show_prediction(infer, args.image_size, args.image_predict)
	


if __name__ == "__main__":

    main()

    print("=== Finished ===")