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
import os
import argparse
import time

# Libs
import numpy as np
import pandas as pd

# If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import tensorflow as tf

# Own modules
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
parser.add_argument("-out", '--detections_out', default='detections.csv',
                    help='Labelmap path', required=False)
parser.add_argument("-lat", '--latency_out', default="latency.csv", help='Output path for latencies file, which is '
                                                                         'appended or created new. ',
                    required=False)

parser.add_argument("-ms", '--model_short_name', default=None, type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-m", '--model_name', default="Model", type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-hw", '--hardware_name', default="Hardware", type=str,
                    help='Hardware name collecting statistical data.', required=False)

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


# def create_single_imagedict(source, image_name):
#     '''
#
#
#     :type
#
#     '''
#     image_dict = {}
#     image_path = os.path.join(source, image_name)
#     image_np = bbox.load_image_into_numpy_array(image_path)
#     input_tensor = np.expand_dims(image_np, 0)
#     image_dict[image_name] = (image_np, input_tensor)
#     return image_dict


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

def plot_image(image, title=None):
    ax = plt.subplot(111)
    ax.tick_params(labelbottom=False, labelleft=False)
    if title:
        plt.title(title, fontsize=40)
    plt.imshow(image)

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()

def infer_images(model_path, image_dir, labelmap, latency_out, detections_out, min_score, model_name,
                 hardware_name, model_short_name=None):
    '''
    Load a saved model, infer and save detections

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

    # Load inference images
    print("Loading images from ", image_dir)
    image_list = im.get_images_name(image_dir)

    # Load label path
    #print("Loading labelmap from ", labelmap)
    #category_index = label_map_util.create_category_index_from_labelmap(os.path.abspath(labelmap))

    #if run_detection:
    # Load model
    print("Loading model {} from {}".format(model_name, model_path))
    detector = load_model(model_path)
    print("Inference with the model {} on hardware {} will be executed".format(model_name, hardware_name))
    #else:
    #    # Load stored XML Files
    #    print("Loading saved Detections files from ealier inferences from ", xml_dir)
    #    data = pd.read_csv(os.path.join(xml_dir, "detections.csv"), sep=';').set_index('filename')

    # Define scores and latencies
    #latencies = pd.DataFrame(columns=['Network', 'Hardware', 'Latency'])
    latencies = []
    detection_scores = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin',
                                             'ymin', 'xmax', 'ymax', 'score'])
    # Process each image
    for image_name in image_list:

        #if run_detection:
        image_filename, image_np, boxes, classes, scores, latency = \
            detect_image(detector, os.path.join(image_dir, image_name))

        latencies.append(latency)
        #latencies=latencies.append(pd.DataFrame([[model_name, hardware_name, latency]], columns=['Network', 'Hardware', 'Latency']))
        bbox_df = convert_reduced_detections_to_df(image_filename, image_np, boxes, classes, scores, min_score)
        detection_scores=detection_scores.append(bbox_df)

    #Save all detections
    #if run_detection and xml_dir and detection_scores.shape[0] > 0:
    # Save detections
    detection_scores.to_csv(detections_out, index=None, sep=';')
    print("Detections saved to ", detections_out)

    if len(latencies) > 1:
        latencies.pop(0)
        print("Removed the first inference time value as it usually includes a warm-up phase. Size of old list: {}. "
              "Size of new list: {}".format(len(latencies) + 1, len(latencies)))

    batch_size=1
    number_runs=len(latencies)

    save_latencies_to_csv(latencies, batch_size, number_runs, hardware_name, model_name, model_short_name, latency_out)

    #print("Saved latency values to ", latency_out)


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
    #throughput = 1 / mean_latency
    throughput = number_runs * batch_size / latencies.sum()

    # Save latencies
    print("Mean inference time: {}".format(mean_latency))
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
    if (len(model_name.split("_", 4))>4):
        custom_parameters = model_name.split("_", 4)[4]
    else:
        custom_parameters = ""
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

        merged_df = old_df.reset_index().merge(df.reset_index(), how="outer").set_index('Date').drop(
            columns=['index'])  # pd.merge(old_df, df, how='outer')

        merged_df.to_csv(latency_out, mode='w', header=True, sep=';')
        # df.to_csv(latency_out, mode='a', header=False, sep=';')
        print("Appended evaluation to ", latency_out)
    else:
        df.to_csv(latency_out, mode='w', header=True, sep=';')
        print("Created new measurement file ", latency_out)

if __name__ == "__main__":
    infer_images(args.model_path, args.image_dir, args.labelmap, args.latency_out, args.detections_out, args.min_score,
                 args.model_name, args.hardware_name, model_short_name=args.model_short_name)

    print("=== Program end ===")
