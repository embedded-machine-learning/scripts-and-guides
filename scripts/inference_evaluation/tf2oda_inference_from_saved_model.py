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
parser.add_argument("-p", '--model_path', default='pre-trained-models/efficientdet_d5_coco17_tpu-32/saved_model/',
                    help='Saved model path', required=False)
parser.add_argument("-i", '--image_dir', default='images/inference',
                    help='Saved model path', required=False)
parser.add_argument("-l", '--labelmap', default='annotations/mscoco_label_map.pbtxt.txt',
                    help='Labelmap path', required=False)
parser.add_argument("-s", '--min_score', default=0.5, type=float,
                    help='Max score of detection box to save the image.', required=False)

parser.add_argument("-r", '--run_detection', default=False,
                    help='Run detection or load saved detection model', required=False, type=bool)
parser.add_argument("-x", '--xml_dir', default=None,
                    help='Source of XML files. '
                         'If run_detection is True, xml files are saved here. '
                         'If run detection is False, XML files are loaded from here. '
                         'If run_detection is True and value is None, no XMLs are saved.', required=False, type=str)
parser.add_argument("-vis", '--run_visualization', default=False,
                    help='Run image visualization', required=False, type=bool)
parser.add_argument("-o", '--output_dir', default="detection_images", help='Result directory for images. ',
                    required=False)
parser.add_argument("-lat", '--latency_out', default="latency.csv", help='Output path for latencies file, which is '
                                                                         'appended or created new. ',
                    required=False)

parser.add_argument("-m", '--model_name', default="Model", type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-hw", '--hardware_name', default="Hardware", type=str,
                    help='Hardware name collecting statistical data.', required=False)

args = parser.parse_args()


# def load_image_into_numpy_array(path):
#   """Load an image from file into a numpy array.
#
#   Puts image into numpy array to feed into tensorflow graph.
#   Note that by convention we put it into a numpy array with shape
#   (height, width, channels), where channels=3 for RGB.
#
#   Args:
#     path: a file path (this can be local or on colossus)
#
#   Returns:
#     uint8 numpy array with shape (img_height, img_width, 3)
#   """
#   img_data = tf.io.gfile.GFile(path, 'rb').read()
#   image = Image.open(BytesIO(img_data))
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)


# def load_labelmap(path):
#     '''
#
#     :param path:
#     :return:
#     '''
#
#     #labelmap = label_map_util.load_labelmap(path)
#     category_index = label_map_util.create_category_index_from_labelmap(path)
#
#     return category_index

# def make_windows_path(path):
#    '''#


#    :param path:
#    :return:
#    '''
# Select paths based on OS
# if (os.name == 'nt'):
#    print("Windows system")
# Windows paths
# mo_file = os.path.join("C://", "Program Files (x86)", "IntelSWTools", "openvino", "deployment_tools",
#                       "model_optimizer", "mo.py")

#    new_path = os.path.join(path)

# path = path.strip() #replace("//", "/")
# else:
#    new_path = path

#    if not (os.path.isdir(path) or os.path.isfile(path)):
#        print("File or folder does not exist. Add current path")
#        new_path = os.path.join(os.getcwd(), new_path)

#    return new_path

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


# def load_images(source):
#     '''
#
#
#     :param source:
#     :return:
#     '''
#     #print(source)
#
#     source = source.replace('\\', '/')
#     image_names = [f for f in os.listdir(source)
#               if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]
#
#     #print(image_names)
#     #input()
#     image_dict = dict()
#     for image_name in image_names:
#         image_path = os.path.join(source, image_name)
#         image_np = bbox.load_image_into_numpy_array(image_path)
#         input_tensor = np.expand_dims(image_np, 0)
#         image_dict[image_name] = (image_np, input_tensor)
#
#         #plt.imshow(image_np)
#         #plt.show()
#
#     return image_dict

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
            xml_df=xml_df.append(pd.DataFrame([content], columns=column_name))

    return xml_df

    # for image_name, value in image_dict.items():
    #     image_np, input_tensor = value
    #     start_time = time.time()
    #     detections = detect_fn(input_tensor)
    #     end_time = time.time()
    #     diff = end_time - start_time
    #     elapsed.append(diff)
    #     print("Inference time {} : {}s".format(image_name, diff))
    #
    #     detections['detection_boxes'][0].numpy(),
    #     detections['detection_classes'][0].numpy().astype(np.int32),
    #     detections['detection_scores'][0].numpy(),
    #
    #     detection_dict[image_name] = (image_np, detections['detection_boxes'][0].numpy(),
    #                                   detections['detection_classes'][0].numpy().astype(np.int32),
    #                                   detections['detection_scores'][0].numpy())

    # mean_elapsed = sum(elapsed) / float(len(elapsed))
    # print('Mean elapsed time: ' + str(mean_elapsed) + 's/image')

    # return detections


# def convert_df_to_reduced_detections(df):
#     '''
#
#
#
#     '''
#
# def visualize_image(image_filename, image_np, boxes, classes, scores, category_index, output_dir, min_score=0.5):
#     '''
#
#     :param detection_array:
#     :param category_index:
#     :return:
#     '''
#
#     image_dir = output_dir
#     if os.path.isdir(image_dir) == False:
#         os.makedirs(image_dir)
#         print("Created directory {}".format(image_dir))
#
#     # print("Visualize images")
#
#     # for image_name, value in detection_array.items():
#     # Get objects
#     # image_np, boxes, classes, scores = value
#
#     if (max(scores) >= min_score):
#         print(image_filename)
#         # print(value)
#         # print(classes)
#         # print(scores)
#         # print(boxes)
#         # input()
#
#         plt.rcParams['figure.figsize'] = [42, 21]
#         label_id_offset = 1
#         image_np_with_detections = image_np.copy()
#         viz_utils.visualize_boxes_and_labels_on_image_array(
#             image_np_with_detections,
#             boxes,
#             classes,
#             scores,
#             category_index,
#             use_normalized_coordinates=True,
#             max_boxes_to_draw=200,
#             min_score_thresh=min_score,
#             agnostic_mode=False)
#         # plt.show()
#         # plt.subplot(5, 1, 1)
#         plt.gcf()
#         plt.imshow(image_np_with_detections)
#
#         new_image_path = os.path.join(image_dir, image_filename + "_det" + ".png")
#         print("Save image {} to {}".format(image_filename, new_image_path))
#         plt.savefig(new_image_path)

def plot_image(image, title=None):
    ax = plt.subplot(111)
    ax.tick_params(labelbottom=False, labelleft=False)
    if title:
        plt.title(title, fontsize=40)
    plt.imshow(image)

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()

def infer_images(model_path, image_dir, labelmap, output_dir, min_score, run_detection, run_visualization, model_name,
                 hardware_name, latency_out, xml_dir=None):
    '''


    '''
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Created ", output_dir)

    if not os.path.isdir(xml_dir):
        os.makedirs(xml_dir)
        print("Created ", xml_dir)

    # Load inference images
    print("Loading images from ", image_dir)
    image_list = im.get_images_name(image_dir)

    # Load label path
    print("Loading labelmap from ", labelmap)
    category_index = label_map_util.create_category_index_from_labelmap(os.path.abspath(labelmap))

    if run_detection:
        # Load model
        print("Loading model {} from {}".format(model_name, model_path))
        detector = load_model(model_path)
        print("Inference with the model {} on hardware {} will be executed".format(model_name, hardware_name))
    else:
        # Load stored XML Files
        print("Loading saved Detections files from ealier inferences from ", xml_dir)
        data = pd.read_csv(os.path.join(xml_dir, "detections.csv"), sep=';').set_index('filename')

    # Define scores and latencies
    #latencies = pd.DataFrame(columns=['Network', 'Hardware', 'Latency'])
    latencies = []
    detection_scores = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin',
                                             'ymin', 'xmax', 'ymax', 'score'])
    # Process each image
    for image_name in image_list:

        if run_detection:
            image_filename, image_np, boxes, classes, scores, latency = \
                detect_image(detector, os.path.join(image_dir, image_name), min_score)

            latencies.append(latency)
            #latencies=latencies.append(pd.DataFrame([[model_name, hardware_name, latency]], columns=['Network', 'Hardware', 'Latency']))
            bbox_df = convert_reduced_detections_to_df(image_filename, image_np, boxes, classes, scores, min_score)
            detection_scores=detection_scores.append(bbox_df)
        else:
            print("Load xml data for that image")
            classes = np.array(data.loc[image_name]['class'])
            scores = np.array(data.loc[image_name]['score'])
            boxes = np.array(data.loc[image_name][['ymin', 'xmin', 'ymax', 'xmax']])
            image_filename = image_name
            image_np = im.load_image_into_numpy_array(os.path.join(image_dir, image_name))


        # If output directory is provided, visualize and save image
        if run_visualization and output_dir:
            image = bbox.visualize_image(image_name, image_np, scores, boxes, classes, category_index, min_score=min_score)

            plt.gcf()
            #plt.imshow(image_np_with_detections)
            new_image_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + "_det" + ".png")
            print("Save image {} to {}".format(image_filename, new_image_path))

            fig = plot_image(image)
            plt.savefig(new_image_path)

    #Save all detections
    if run_detection and xml_dir and detection_scores.shape[0] > 0:
        # Save detections
        detection_scores.to_csv(os.path.join(xml_dir, "detections.csv"), index=None, sep=';')
        print("Detections saved to ", os.path.join(xml_dir, "detections.csv"))

        # Save latencies
        # Calucluate mean latency
        if len(latencies) > 1:
            latencies.pop(0)
            print("Removed the first inference time value as it usually includes a warm-up phase. Size of old list: {}. "
                  "Size of new list: {}".format(len(latencies)+1, len(latencies)))

        mean_latency = np.array(latencies).mean()
        print("Mean inference time: ".format(mean_latency))

        series_index = ['Date',
                        'Network',
                        'Hardware',
                        'MeanLatency',
                        'Latencies']

        content = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, hardware_name, mean_latency]
        content.append(latencies)

        # Create DataFrame
        df = pd.DataFrame([pd.Series(data=content, index=series_index, name="data")])
        df.set_index('Date', inplace=True)

        # Append dataframe wo csv if it already exists, else create new file
        if os.path.isfile(latency_out):
            df.to_csv(latency_out, mode='a', header=False, sep=';')
            print("Appended evaluation to ", latency_out)
        else:
            df.to_csv(latency_out, mode='w', header=True, sep=';')
            print("Created new measurement file ", latency_out)

        #Save inferences
        #latencies_path = os.path.join(output_dir, "latencies.csv")
        #latencies.to_csv(latency_out, sep=';')
        print("Saved latency values to ", latency_out)


if __name__ == "__main__":
    infer_images(args.model_path, args.image_dir, args.labelmap, args.output_dir, args.min_score, args.run_detection,
                 args.run_visualization, args.model_name, args.hardware_name, args.latency_out, args.xml_dir)

    print("=== Program end ===")
