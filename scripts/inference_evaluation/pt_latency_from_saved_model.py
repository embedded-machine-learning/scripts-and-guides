#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Infer pytorch models for latency

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
import logging

# Libs
import numpy as np
import cv2

import torch
from torchvision import datasets, transforms
#import helper

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

parser = argparse.ArgumentParser(description='Pytorch YoloV5 Latency Inferrer')
parser.add_argument("-p", '--model_path', default='C:\Projekte\21_SoC_EML\eml_projects\yolov5-oxford-pets',
                    help='Saved model path', required=False)
parser.add_argument("-i", '--image_dir', default='images/inference',
                    help='Images', required=False)
parser.add_argument("-l", '--labelmap', default='annotations/mscoco_label_map.pbtxt.txt',
                    help='Labelmap path', required=False)
parser.add_argument("-s", '--min_score', default=0.5, type=float,
                    help='Max score of detection box to save the image.', required=False)
#parser.add_argument("-out", '--detections_out', default='detections.csv',
#                    help='Output file detections', required=False)
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

def infer_latency_images(model_path, image_dir, latency_out, model_name,
                 hardware_name, model_short_name=None, batch_size=1, image_size=None,
                 model_optimizer_prefix='TRT', index_save_file="./tmp/index.txt", N_warmup_run=50, N_run=1000):
    """


    """

    # Create output directories
    os.makedirs(os.path.dirname(latency_out), exist_ok=True)
    os.makedirs(os.path.dirname(index_save_file), exist_ok=True)

    # Get model infos
    model_info = inf.get_info_from_modelname(model_name, model_short_name,
                                             model_optimizer_prefix=model_optimizer_prefix)

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

    # Convert image to numpy array
    img = cv2.imread(os.path.join(image_dir, image_list[0]))
    # Get the batch input from here
    res = cv2.resize(img, dsize=(image_size[0], image_size[1]), interpolation=cv2.INTER_CUBIC)

    # Model
    #model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo
    model = torch.hub.load('./', 'custom', force_reload=True, source='local', path=model_path)

    elapsed_time = []

    print("Running warm up runs...i.e. just running empty runs to load the model correctly")
    for i in range(N_warmup_run):
        results = model(res)
        # print("Inference {}/{}".format(i, N_warmup_run))
        # preds = labeling['predictions'].numpy()

    print("Running real runs with one batch to create the images...")
    for i in range(N_run):
        start_time = time.time()
        results = model(res)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # in ms

        elapsed_time.append(latency)

        # Results
        #results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i + 50, (np.array(elapsed_time[-50:]).mean())))

    index = inf.generate_measurement_index(model_name)
    inf.save_latencies_to_csv(elapsed_time, batch_size, N_run, hardware_name, model_name, model_short_name, latency_out,
                          index)
    #Save index to a file
    file1 = open(index_save_file, 'w')
    file1.write(index)
    print("Index {} used for latency measurement".format(index))



if __name__ == "__main__":
    infer_latency_images(args.model_path, args.image_dir, args.latency_out,
                 args.model_name, args.hardware_name, model_short_name=args.model_short_name,
                 batch_size=args.batch_size, image_size=args.image_size,
                 model_optimizer_prefix=args.model_optimizer_prefix, index_save_file=args.index_save_file)

    print("=== Program end ===")