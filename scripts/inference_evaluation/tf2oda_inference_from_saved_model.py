#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Infer TF2 object detection models on images either directly from the model or by loading presaved xml files.

License_info:
# ==============================================================================
# ISC License (ISC)
# Copyright 2021 Christian Doppler Laboratory for Embedded Machine Learning
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

# The following is a slightly modified version from the following script
# Source:

"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import json
import os
import argparse
import time
import warnings
from datetime import datetime
import logging

# Libs
import numpy as np
import pandas as pd

# If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Own modules
import image_utils as im
import inference_utils as inf

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(description='Google Tensorflow Detection API 2.0 Inferrer')
parser.add_argument("-p", '--model_path', default='pre-trained-models/efficientdet_d5_coco17_tpu-32/saved_model/',
                    help='Saved model path', required=False)
parser.add_argument("-i", '--image_dir', default='images/inference',
                    help='Saved model path', required=False)
parser.add_argument("-l", '--labelmap', default='annotations/mscoco_label_map.pbtxt.txt',
                    help='Labelmap path', required=False)
parser.add_argument("-s", '--min_score', default=0.5, type=float,
                    help='Max score of detection box to save the image.', required=False)
parser.add_argument("-out", '--detections_out', default='detections.csv',
                    help='Output file detections', required=False)
parser.add_argument("-lat", '--latency_out', default="latency.csv", help='Output path for latencies file, which is '
                                                                         'appended or created new. ', required=False)

parser.add_argument('-b', '--batch_size', type=int, default=1,
                    help='Batch Size', required=False)
parser.add_argument('-is', '--image_size', type=str, default=None,
                    help='List of two coordinates: [Height, Width]', required=False)

parser.add_argument("-ms", '--model_short_name', default=None, type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-m", '--model_name', default="Model", type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-hw", '--hardware_name', default="Hardware", type=str,
                    help='Hardware name collecting statistical data.', required=False)

parser.add_argument('-mop', '--model_optimizer_prefix', type=str, default='TRT',
                    help='Prefix for Model Optimizer Settings', required=False)
parser.add_argument('-id', '--index_save_file', type=str, default='./tmp/index.txt',
                    help='Path to put index file to keep the same key for different types of measurements.',
                    required=False)
args = parser.parse_args()

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

log.info(args)
#print(args)


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
        pics = data_path
    else:
        pics = os.listdir(data_path)
    n = len(pics)

    for i in range(batch_size):
        if os.path.isfile(data_path):
            img_path = data_path
        else:
            img_path = os.path.join(data_path, pics[i % n])  # generating batches
        img = image.load_img(img_path, target_size=(hw[0], hw[1]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if is_keras:
            x = preprocess_input(x)  # for models loaded from Keras applications
        batched_input[i, :] = x

    batched_input = tf.constant(batched_input)
    return batched_input


def infer_latency(infer, image_dir, hardware_name, model_name, model_short_name, latency_out,
                  N_warmup_run=50, N_run=1000, batch_size=1, d_type='uint8', image_size=[300, 300],
                  index_save_file="./tmp/index.txt"):
    '''



    '''

    input = batch_input(batch_size, image_dir, d_type, image_size, is_keras=False)

    elapsed_time = []
    # all_preds = []
    # boxes = []
    # classes = []
    # scores = []
    # batch_size = batched_input.shape[0]

    print("Running warm up runs...i.e. just running empty runs to load the model correctly")
    for i in range(N_warmup_run):
        labeling = infer(input)
        # print("Inference {}/{}".format(i, N_warmup_run))
        # preds = labeling['predictions'].numpy()
        preds = labeling

    print("Running real runs with one batch to create the images...")
    for i in range(N_run):
        start_time = time.time()
        labeling = infer(input)
        # preds = labeling['predictions'].numpy()
        preds = labeling
        end_time = time.time()

        latency = (end_time - start_time) * 1000    #in ms

        elapsed_time.append(latency)
        # elapsed_time = np.append(elapsed_time, end_time - start_time)

        # all_preds.append(preds)

        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i + 50, (np.array(elapsed_time[-50:]).mean()) * 1000))

    # throughput = N_run * batch_size / elapsed_time.sum()

    # Create the latency.csv file
    # Generate identifier for this run
    index = inf.generate_measurement_index(model_name)
    inf.save_latencies_to_csv(elapsed_time, batch_size, N_run, hardware_name, model_name, model_short_name, latency_out,
                          index)
    #Save index to a file
    file1 = open(index_save_file, 'w')
    file1.write(index)
    print("Index {} used for latency measurement".format(index))





def load_model(model_path):
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

def detect_image(detect_fn, image_path):
    '''


    :param detect_fn:
    :param image_dict:

    :return:
    '''
    elapsed = []
    detection_dict = dict()

    # print("Start detection")
    # for image_name in image_list:
    # Load image
    # image_path = os.path.join(image_dir, image_name)
    # Convert image to array
    print("Process ", image_path)
    total_latency_start_time = time.time()
    image_np = im.load_image_into_numpy_array(image_path)

    # Make image tensor of it
    input_tensor = np.expand_dims(image_np, 0)

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

    total_latency_stop_time = time.time()
    total_latency = total_latency_stop_time - total_latency_start_time
    print("Total latency for {} : {}s".format(image_filename, total_latency))

    return image_filename, image_np, boxes, classes, scores, latency, total_latency


def infer_images(model_path, image_dir, latency_out, detections_out, min_score, model_name,
                 hardware_name, model_short_name=None, batch_size=1, image_size=None,
                 model_optimizer_prefix='TRT', index_save_file="./tmp/index.txt"):
    '''
    Load a saved model, infer and save detections

    '''
    # Create output directories
    if not os.path.isdir(os.path.dirname(detections_out)):
        os.makedirs(os.path.dirname(detections_out))
        print("Created ", os.path.dirname(detections_out))

    if not os.path.isdir(os.path.dirname(latency_out)):
        os.makedirs(os.path.dirname(latency_out))
        print("Created ", os.path.dirname(latency_out))

    os.makedirs(os.path.dirname(index_save_file), exist_ok=True)

    # Get model infos
    model_info = inf.get_info_from_modelname(model_name, model_short_name, model_optimizer_prefix=model_optimizer_prefix)
    print("Model information: ", model_info)
    if image_size:
        image_size = json.loads(image_size)
        # image_size = list(map(int, image_size))
        if (image_size[0] != model_info['resolution'][0]) or (image_size[1] != model_info['resolution'][1]):
            warnings.warn("Provided input resolution differs from model resolution: "
                          "Input={}, model={}".format(image_size, model_info['resolution']))
        else:
            print("Using image resolution {}".format(image_size))

    else:
        image_size = model_info['resolution']
        print("In the batch processing, model resolution {} will be used".format(image_size))

    # Load inference images
    print("Loading images from ", image_dir)
    image_list = im.get_images_name(image_dir)

    # Load model
    print("Loading model {} from {}".format(model_name, model_path))
    detector = load_model(model_path)
    print("Inference with the model {} on hardware {} will be executed".format(model_name, hardware_name))

    print("Perform latency tests.")
    infer_latency(detector, image_dir, hardware_name, model_name, model_info['model_short_name'], latency_out,
                  N_warmup_run=50, N_run=1000, batch_size=batch_size, d_type='uint8',
                  image_size=image_size, index_save_file=index_save_file)

    # Define scores and latencies
    latencies = []
    total_latencies = []
    detection_scores = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin',
                                             'ymin', 'xmax', 'ymax', 'score'])
    # Process each image
    for image_name in image_list:
        # if run_detection:
        image_filename, image_np, boxes, classes, scores, latency, total_latency = \
            detect_image(detector, os.path.join(image_dir, image_name))

        latencies.append(latency)
        total_latencies.append(total_latency)
        # latencies=latencies.append(pd.DataFrame([[model_name, hardware_name, latency]], columns=['Network', 'Hardware', 'Latency']))
        bbox_df = inf.convert_reduced_detections_tf2_to_df(image_filename, image_np, boxes, classes, scores, min_score)
        detection_scores = detection_scores.append(bbox_df)

    print("Mean latency without batch processing: {}".format(np.array(latencies[1:-1]).mean()))
    print("Mean total latency including image preprocessing: {}".format(np.array(total_latencies[1:-1]).mean()))


    # Save all detections
    # if run_detection and xml_dir and detection_scores.shape[0] > 0:
    # Save detections
    detection_scores.to_csv(detections_out, index=None, sep=';')
    print("Detections saved to ", detections_out)


if __name__ == "__main__":
    infer_images(args.model_path, args.image_dir, args.latency_out, args.detections_out, args.min_score,
                 args.model_name, args.hardware_name, model_short_name=args.model_short_name,
                 batch_size=args.batch_size, image_size=args.image_size,
                 model_optimizer_prefix=args.model_optimizer_prefix, index_save_file=args.index_save_file)

    print("=== Program end ===")
