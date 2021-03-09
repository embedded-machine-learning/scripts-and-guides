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
import json
import os
import time
import argparse

# Libs
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import re
import tensorflow as tf
from six import BytesIO
from PIL import Image
from tensorflow import keras
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from datetime import datetime

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

parser.add_argument('-mp', '--model_path', default='./inceptionv3_saved_model_TFTRT_FP16', help='Saved optimized model path',
                    required=False)
					
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Batch Size', required=False)
					
parser.add_argument('-is', '--image_size', type=str, default='[300, 300]',
                    help='List of two coordinates: [Height, Width]', required=False)
					
#parser.add_argument('-p', '--precision', default='INT8',
#                    help='TensorRT precision mode: FP32, FP16 or INT8.', required=False)

parser.add_argument('-e', '--dtype', default='uint8',
                    help='Data type for the input from float32, float16 or uint8.', required=False)

parser.add_argument("-s", '--min_score', default=0.5, type=float,
                    help='Max score of detection box to save the image.', required=False)

parser.add_argument("-i", '--image_dir', default='images/inference',
                    help='Saved model path', required=False)
					
parser.add_argument("-lat", '--latency_out', default="latency.csv", help='Output path for latencies file, which is '
                                                                         'appended or created new. ',
                    required=False)

parser.add_argument("-out", '--detections_out', default='detections.csv',
                    help='Labelmap path', required=False)

parser.add_argument("-ms", '--model_short_name', default=None, type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-m", '--model_name', default="Model", type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-hw", '--hardware_name', default="Hardware", type=str,
                    help='Hardware name collecting statistical data.', required=False)
					
args, unknown = parser.parse_known_args()
print(args)


def batch_input(batch_size, data_path, d_type, hw, is_keras=False):
    '''
    Create one representative batch out of the dataset


    TODO: Use all images in the dataset to create a batch, not only the first image, i.e. create
    TODO: a batch as in reality

    '''

    if d_type == 'float32':
        datatype = np.float32
    elif d_type == 'float16':
        datatype = np.float16
    elif d_type == 'uint8':
        datatype = np.uint8
    else:
        raise ValueError("No valid data type provided: " + d_type + ". It has to be float32, float16 or uint8")

    batched_input = np.zeros((batch_size, hw[0], hw[1], 3), dtype=datatype)

    if os.path.isfile(data_path):
        pics=data_path
    else:
        pics = os.listdir(data_path)
    n = len(pics)

    for i in range(batch_size):
        if os.path.isfile(data_path):
            img_path=data_path
        else:
            img_path = os.path.join(data_path, pics[i % n]) #generating batches
        img = image.load_img(img_path, target_size=(hw[0], hw[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if is_keras:
            x = preprocess_input(x) #for models loaded from Keras applications
        batched_input[i, :] = x

    batched_input = tf.constant(batched_input)
    return batched_input

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# def load_image_keras_method(img_path, dtype, target_size, is_keras=False):
#     '''
#
#     :argument
#
#     :return
#
#     '''
#     if dtype == 'float32':
#         datatype = np.float32
#     elif dtype == 'float16':
#         datatype = np.float16
#     elif dtype == 'uint8':
#         datatype = np.uint8
#     else:
#         raise ValueError("No valid data type provided: " + dtype + ". It has to be float32, float16 or uint8")
#
#     batched_input = np.zeros((1, target_size[0], target_size[1], 3), dtype=datatype)
#
#     img = image.load_img(img_path, target_size=(target_size[0], target_size[1]))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     if is_keras:
#         x = preprocess_input(x)  # for models loaded from Keras applications
#     batched_input[0, :] = x
#
#     #x = x.astype(args.dtype)
#     x = tf.constant(x)
#
#     return batched_input, img


def save_latencies_to_csv(latencies, batch_size, number_runs, hardware_name, model_name, model_short_name, latency_out):
    '''
    Save a list of latencies to csv file

    :argument


    :return
        None

    '''

    # Calucluate mean latency
    mean_latency = np.array(latencies).mean()

    # Calulate throughput
    # throughput = 1 / mean_latency
    throughput = number_runs * batch_size / latencies.sum()

    # Save latencies
    print("Mean inference time: {}. Throughput: {}".format(mean_latency, throughput))
    series_index = ['Date',
                    'Model',
                    'Model_Short',
                    'Framework',
                    'Network',
                    'Resolution',
                    'Dataset',
                    'Custom_Parameters',
                    'Hardware',
                    'Hardware_Optimization',
                    'Batch_Size',
                    'Throughput',
                    'Mean_Latency',
                    'Latencies']
    framework = str(model_name).split('_')[0]
    network = str(model_name).split('_')[1]
    resolution = str(model_name).split('_')[2]
    dataset = str(model_name).split('_')[3]
    custom_parameters = model_name.split("_", 4)[4]
    content = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               model_name,
               model_short_name,
               framework,
               network,
               resolution,
               dataset,
               custom_parameters,
               hardware_name,
               'None',
               1,
               throughput,
               mean_latency,
               str(latencies)]
    # Create DataFrame
    df = pd.DataFrame([pd.Series(data=content, index=series_index, name="data")])
    df.set_index('Date', inplace=True)
    # Append dataframe wo csv if it already exists, else create new file
    if os.path.isfile(latency_out):
        old_df = pd.read_csv(latency_out, sep=';')
        old_df['Custom_Parameters'] = old_df['Custom_Parameters'].replace(np.nan, '', regex=True)
        
        merged_df = old_df.reset_index().merge(df.reset_index(), how="outer").set_index('Date').drop(
            columns=['index'])  # pd.merge(old_df, df, how='outer')

        merged_df.to_csv(latency_out, mode='w', header=True, sep=';')
        # df.to_csv(latency_out, mode='a', header=False, sep=';')
        print("Appended evaluation to ", latency_out)
    else:
        df.to_csv(latency_out, mode='w', header=True, sep=';')
        print("Created new measurement file ", latency_out)


def infer_latency(batched_input, infer, hardware_name, model_name, model_short_name, latency_out,
                  N_warmup_run=50, N_run=1000):
    elapsed_time = []
    #all_preds = []
    #boxes = []
    #classes = []
    #scores = []
    batch_size = batched_input.shape[0]
    
    print("Running warm up runs...i.e. just running empty runs to load the model correctly")
    for i in range(N_warmup_run):
        labeling = infer(batched_input)
        #print("Inference {}/{}".format(i, N_warmup_run))
        #preds = labeling['predictions'].numpy() 
        preds = labeling

    print("Running real runs with one batch to create the images...")
    for i in range(N_run):
        start_time = time.time()
        labeling = infer(batched_input)
        #preds = labeling['predictions'].numpy() 
        preds = labeling
        end_time = time.time()

        elapsed_time = np.append(elapsed_time, end_time - start_time)

        #all_preds.append(preds)

        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i + 50, (elapsed_time[-50:].mean()) * 1000))
	
    #throughput = N_run * batch_size / elapsed_time.sum()

    #Create the latency.csv file
    save_latencies_to_csv(elapsed_time, batch_size, N_run, hardware_name, model_name, model_short_name, latency_out)



def infer_performance(model, image_dir, d_type, image_size, detections_out, min_score, hardware_name, model_name, model_short_name):
    '''


    :argument


    :return

    '''

    #decoding predictions
    target_hw = json.loads(image_size)

    #detection_boxes = []
    #detection_scores = []
    #detection_classes = []
    #score = []

    latencies = []
    detection_scores = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin',
                                             'ymin', 'xmax', 'ymax', 'score'])

    pics = os.listdir(image_dir)
    for pic in pics:
        #img_path = os.path.join(image_dir, pic)

        image_filename, image_np, boxes, classes, scores, latency = \
            detect_image(model, os.path.join(image_dir, pic), d_type, target_hw)

        #x = load_image_keras_method(img_path, d_type, target_hw)
        #pred = model(x)

        latencies.append(latency)
        # latencies=latencies.append(pd.DataFrame([[model_name, hardware_name, latency]], columns=['Network', 'Hardware', 'Latency']))
        bbox_df = convert_reduced_detections_to_df(image_filename, image_np, boxes, classes, scores, min_score)
        detection_scores = detection_scores.append(bbox_df)

    # Save all detections
    # if run_detection and xml_dir and detection_scores.shape[0] > 0:
    # Save detections
    detection_scores.to_csv(detections_out, index=None, sep=';')
    print("Detections saved to ", detections_out)


    #     detection_boxes.append(pred['detection_boxes'].numpy()[0])
    #     for x in pred['detection_scores'].numpy()[0]:
    #         if x > 0.02:
    #             score.append(x)
    #     detection_scores.append(np.array(score))
    #     detection_classes.append(pred['detection_classes'].numpy()[0])
    #
    #
    # # writing csv file
    # now = datetime.datetime.now()
    # name_list = net.split('_')
    # while len(name_list) < 4:
    #     name_list.append('ND')
    #
    # headers =     ['Date',
    #                 'Model',
    #                 'Model_Short',
    #                 'Framework',
    #                 'Network',
    #                 'Resolution',
    #                 'Dataset',
    #                 'Hardware',
    #                 'Hardware_Optimization',
    #                 'Precision',
    #                 'Batch_Size',
    #                 'Throughput',
    #                 'Mean_Latency',
    #                 'Latencies',
    #                 'detection_boxes',
    #                 'detection_scores',
    #                 'detection_classes']
    #
    # body =        [now.strftime('%Y-%m-%d'),# %H:%M:%S"),
    #                 "MODEL_TBD",
    #                 "MODEL_SHORT_TBD",
    #                 name_list[0],
    #                 name_list[1],
    #                 name_list[2],
    #                 name_list[3],
    #                 "NVIDIA_XAVIER_TBD",
    #                 "TRT_TBD",
    #                 precision,
    #                 batch_size,
    #                 '{:.0f} images/s'.format(throughput),
    #                 '{:.1f} ms'.format(elapsed_time.mean() * 1000),
    #                 "SINGLE_LATENCIES_TBD",
    #                 detection_boxes,
    #                 detection_scores,
    #                 detection_classes]
    #
    # if os.path.exists(csv):
    #     with open(csv, 'a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(body)
    #
    # else:
    #     with open(csv, 'a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(headers)
    #         writer.writerow(body)
    #
    #
    # print('Throughput: {:.0f} images/s'.format(throughput))
    # return all_preds

def convert_reduced_detections_to_df(image_filename, image_np, boxes, classes, scores, min_score=0.8):
    image_width = image_np.shape[1]
    image_height = image_np.shape[0]

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
    xml_df = pd.DataFrame(columns=column_name)

    for i in range(scores.shape[0]):
        if min_score <= scores[i]:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            content = [image_filename, image_width, image_height,
                       classes[i], xmin, ymin, xmax, ymax, scores[i]]
            xml_df=xml_df.append(pd.DataFrame([content], columns=column_name))

    return xml_df


def load_model_keras(input_saved_model_dir):
    '''
    Load trt model

    '''
    print(f'Loading saved model {input_saved_model_dir}...')
    start_time = time.time()
    saved_model_loaded = tf.saved_model.load(input_saved_model_dir, tags=[tag_constants.SERVING])
    end_time = time.time()
    print('Loading model took {:4.1f}s'.format(end_time - start_time))
    return saved_model_loaded

def load_model_default(model_path):
    '''
    Load tensorflow model

    :param model_path:
    :return:
    '''

    print("Start model loading from path ", model_path)
    tf.keras.backend.clear_session()
    start_time = time.time()
    detect_fn = tf.saved_model.load(model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Finished. Elapsed time: {:.0f}s'.format(elapsed_time))

    return detect_fn

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def detect_image(detect_fn, image_path, d_type, target_hw):
    '''


    :param detect_fn:
    :param image_dict:

    :return:
    '''
    #elapsed = []
    #detection_dict = dict()

    # print("Start detection")
    # for image_name in image_list:
    # Load image
    # image_path = os.path.join(image_dir, image_name)
    # Convert image to array
    print("Process ", image_path)
    #image_tensor, image_np = load_image_keras_method(image_path, d_type, target_hw)
    #image_np = load_image_into_numpy_array(image_path)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = batch_input(1, image_path, d_type, target_hw, is_keras=False)
    #input_tensor, image_np = load_image_keras_method(image_path, d_type, target_hw, is_keras=False)

    # Make image tensor of it
    #input_tensor = np.expand_dims(image_np, 0)

    #image_np = load_image_into_numpy_array(image_path)

    # Make image tensor of it
    #input_tensor = np.expand_dims(image_np, 0)

    # Infer
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()

    latency = end_time - start_time
    # elapsed.append(latency)

    image_filename = os.path.basename(image_path)

    print("Inference time {} : {}s".format(image_filename, latency))

    # Process detections
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    return image_filename, image_np, boxes, classes, scores, latency



def infer_images(model, image_size, batch_size, image_dir, dtype, latency_out, detections_out, min_score, model_name,
                 hardware_name, model_short_name):
    '''
    Infer images with tensor-rt

    :argument

    :return

    '''

    #Create output directories
    if not os.path.isdir(os.path.dirname(detections_out)):
        os.makedirs(os.path.dirname(detections_out))
        print("Created ", os.path.dirname(detections_out))

    if not os.path.isdir(os.path.dirname(latency_out)):
        os.makedirs(os.path.dirname(latency_out))
        print("Created ", os.path.dirname(latency_out))

    #Enhance inputs
    if model_short_name is None:
        model_short_name=model_name
        print("No short models name defined. Using the long name: ", model_name)

    #Image size
    target_size = json.loads(image_size)
    print("Using image size ", target_size)

    #making batched inputs
    print("=== Prepare batch input ===")
    batched_input = batch_input(batch_size, image_dir, dtype, target_size, is_keras=False)
    print("=== Batch input prepared ===")

    # Load trt model
    print("=== Load saved model ===")
    saved_model_loaded = load_model_default(model)  #Load trt saved model
    model = saved_model_loaded.signatures['serving_default']
    print("=== Model loaded ===")

    # Latencies
    print("=== Infer latencies ===")
    infer_latency(batched_input, model, hardware_name, model_name,
                  model_short_name, latency_out, N_warmup_run=50, N_run=1000)

    # Performance
    print("=== Infer performance of selected images ===")
    infer_performance(model, image_dir, dtype, image_size, detections_out, min_score, hardware_name,
                      model_name, model_short_name)


    #all_preds = predict_and_benchmark_throughput(batched_input, infer, args, N_warmup_run=50, N_run=1000)
    print("=== Inference complete ===")
    #if (args.visualize):
    #    show_prediction(infer, args.image_size, args.image_predict)
	


if __name__ == "__main__":

    infer_images(args.model_path, args.image_size, args.batch_size, args.image_dir, args.dtype, args.latency_out,
                 args.detections_out, args.min_score, args.model_name, args.hardware_name,
                 model_short_name=args.model_short_name)

    print("=== Finished ===")