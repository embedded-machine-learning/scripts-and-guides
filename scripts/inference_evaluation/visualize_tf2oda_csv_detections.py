#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize detections made by tensorflow.

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
import glob
import json
import os
import argparse
import time
import re
import pickle

# Libs
from tqdm import tqdm
from xmltodict import unparse
import xml.etree.ElementTree as ET
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

import matplotlib

# If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
matplotlib.use('TkAgg')

from six import BytesIO

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Own modules
import bbox_utils as bbox
import image_utils as im
from datetime import datetime

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
# parser.add_argument("-p", '--model_path', default='pre-trained-models/efficientdet_d5_coco17_tpu-32/saved_model/',
#                    help='Saved model path', required=False)
parser.add_argument("-i", '--image_dir', default='images/inference',
                    help='Image directory', required=False)
parser.add_argument("-l", '--labelmap', default='annotations/mscoco_label_map.pbtxt.txt',
                    help='Labelmap path', required=False)
parser.add_argument("-d", '--detections_file', default='detections.csv',
                    help='TF2 Object Detection API saved inference file as csv.', required=False)
parser.add_argument("-s", '--min_score', default=0.5, type=float,
                    help='Max score of detection box to save the image.', required=False)

# parser.add_argument("-r", '--run_detection', default=False,
#                    help='Run detection or load saved detection model', required=False, type=bool)
# parser.add_argument("-x", '--xml_dir', default=None,
#                    help='Source of XML files. '
#                         'If run_detection is True, xml files are saved here. '
#                         'If run detection is False, XML files are loaded from here. '
#                         'If run_detection is True and value is None, no XMLs are saved.', required=False, type=str)
# parser.add_argument("-vis", '--run_visualization', default=False,
#                    help='Run image visualization', required=False, type=bool)
parser.add_argument("-o", '--output_dir', default="detection_images", help='Result directory for images. ',
                    required=False)
# parser.add_argument("-lat", '--latency_out', default="latency.csv", help='Output path for latencies file, which is '
#                                                                         'appended or created new. ',
#                    required=False)

# parser.add_argument("-m", '--model_name', default="Model", type=str,
#                    help='Model name for collecting model data.', required=False)
# parser.add_argument("-hw", '--hardware_name', default="Hardware", type=str,
#                    help='Hardware name collecting statistical data.', required=False)

args = parser.parse_args()


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


def create_single_imagedict(source, image_name):
    '''


    :type

    '''
    image_dict = {}
    image_path = os.path.join(source, image_name)
    image_np = bbox.load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    image_dict[image_name] = (image_np, input_tensor)
    return image_dict


def detect_image(detect_fn, image_path, min_score):
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

    return image_filename, image_np, boxes, classes, scores, latency


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
            xml_df = xml_df.append(pd.DataFrame([content], columns=column_name))

    return xml_df


def plot_image(image, title=None):
    ax = plt.subplot(111)
    ax.tick_params(labelbottom=False, labelleft=False)
    if title:
        plt.title(title, fontsize=40)
    plt.imshow(image)

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()


def infer_images(detections_file, image_dir, labelmap, min_score, output_dir):
    '''


    '''
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Created ", output_dir)

    # Load inference images
    print("Loading images from ", image_dir)
    image_list = im.get_images_name(image_dir)

    # Load label path
    print("Loading labelmap from ", labelmap)
    category_index = label_map_util.create_category_index_from_labelmap(os.path.abspath(labelmap))

    # Load stored XML Files
    print("Loading saved Detections files from ealier inferences from ", detections_file)
    data = pd.read_csv(detections_file, sep=';').set_index('filename')

    # Define scores and latencies
    # detection_scores = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin',
    #
    # 'ymin', 'xmax', 'ymax', 'score'])
    # Process each image
    for image_name in image_list:

        print("Load xml data for ", image_name)
        if isinstance(data.loc[image_name], pd.Series):
            subdata = pd.DataFrame([data.loc[image_name]])
        else:
            subdata = data.loc[image_name]
        #classes = np.array(data.loc[image_name].shape[0])
        boxes = np.zeros([subdata.shape[0], 4])
        classes = np.zeros([subdata.shape[0]]).astype('int')

        if 'score' in subdata.columns and subdata['score'][0] is not None:
            scores = np.zeros([subdata.shape[0]])
        else:
            scores = None

        for i in range(subdata.shape[0]):
            boxes[i][0] = subdata['ymin'][i]
            boxes[i][1] = subdata['xmin'][i]
            boxes[i][2] = subdata['ymax'][i]
            boxes[i][3] = subdata['xmax'][i]

            if 'score' in data.columns and subdata['score'][i] is not None:
                scores[i] = subdata['score'][i]

            #class_index = [category_index[j + 1].get('id') for j in range(len(category_index)) if
            #               category_index[j + 1].get('name') == subdata['class'][i]][0]

            classes[i] = subdata['class'][i]

        #if classes.size == 1:  # If only one detection
        #    classes = classes.reshape(-1)
        #    np.vstack(classes, 1)
        #scores = np.array(data.loc[image_name]['score'])
        #if scores.size == 1:  # If only one detection
        #    scores = scores.reshape(-1)
        #    np.vstack(scores, 0)
        #boxes = np.array(data.loc[image_name][['ymin', 'xmin', 'ymax', 'xmax']])
        #if boxes.size==4:
        #    boxes = boxes.reshape(1, -1)
        #    np.vstack(boxes, [0, 0, 0, 0])
        image_filename = image_name
        image_np = im.load_image_into_numpy_array(os.path.join(image_dir, image_name))

        # If output directory is provided, visualize and save image
        image = bbox.visualize_image(image_name, image_np, scores, boxes, classes, category_index, min_score=min_score)

        plt.gcf()
        new_image_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + "_det" + ".png")
        print("Save image {} to {}".format(image_filename, new_image_path))

        fig = plot_image(image)
        plt.savefig(new_image_path)


if __name__ == "__main__":
    infer_images(args.detections_file, args.image_dir, args.labelmap, args.min_score, args.output_dir)

    print("=== Program end ===")
