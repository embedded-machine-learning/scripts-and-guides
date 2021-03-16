#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Select two or three images with their bounding boxes in PASCAL VOC XML format and visualize then within one image.

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

# The following script uses several method fragments from Tensorflow
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

Tensorflow has the following licence:
# ==============================================================================
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
import time

# Libs
import argparse
import numpy as np
import glob
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import matplotlib
from six import BytesIO
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tkinter

# Own modules
import bbox_utils as bbox

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

#If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(description='Google Tensorflow Detection API 2.0 Inferrer')
parser.add_argument('--labelmap', default='annotations/label_map.pbtxt',
                    help='Labelmap path', required=False)
parser.add_argument('--output_dir', default="samples/result",
                    help='Result directory', required=False)

parser.add_argument("--image_path1", type=str, default="images/0.jpg", help='Path to image, usually ground truth',
                    required=False)
parser.add_argument("--image_path2", type=str, default="None", help='Path to image, usually some model to compare with. '
                    'If not set, path of image1 will be used', required=False)
parser.add_argument("--image_path3", type=str, default="None", help='Path to image, usually some model to compare with. '
                    'If not set, path of image1 will be used', required=False)

parser.add_argument("--annotation_dir1", type=str, default="annotations/xmls", help='Path to xml with bounding boxes.',
                    required=False)
parser.add_argument("--annotation_dir2", type=str, default="None", help='Path to xml with bounding boxes. If not set, '
                    'then, the value from annotation_dir1 will be used', required=False)
parser.add_argument("--annotation_dir3", type=str, default="None", help='Path to xml with bounding boxes. If not set, '
                    'then, the value from annotation_dir1 will be used', required=False)

parser.add_argument("--title1", type=str, default="Image1", help='Title of image 1', required=False)
parser.add_argument("--title2", type=str, default="Image2", help='Title of image 2', required=False)
parser.add_argument("--title3", type=str, default="Image3", help='Title of image 3', required=False)

parser.add_argument("--use_three_images", action='store_true', default=False,
                    help="If set, three images will be used, instead of two")
parser.add_argument("--color_gt", action='store_true', default=False,
                    help="Color ground truth and set confidence to 100%")

args = parser.parse_args()
print(args)


def load_labelmap(path):
    '''

    :param path:
    :return:
    '''

    #labelmap = label_map_util.load_labelmap(path)
    category_index = label_map_util.create_category_index_from_labelmap(path)

    return category_index

def load_model(model_path):
    '''


    :param model_path:
    :return:
    '''
    start_time = time.time()
    print("Start model loading from path ", model_path)
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Finished. Elapsed time: ' + str(elapsed_time) + 's')

    return detect_fn


# def get_images_name(source):
#     source = source.replace('\\', '/')
#     image_names = [f for f in os.listdir(source)
#               if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]
#     return image_names

# def create_single_imagedict(source,image_name):
#     image_dict = {}
#     image_path = os.path.join(source, image_name)
#     image_np = load_image_into_numpy_array(image_path)
#     input_tensor = np.expand_dims(image_np, 0)
#     image_dict[image_name] = (image_np, input_tensor)
#     return image_dict


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
#         image_np = load_image_into_numpy_array(image_path)
#         input_tensor = np.expand_dims(image_np, 0)
#         image_dict[image_name] = (image_np, input_tensor)
#
#         #plt.imshow(image_np)
#         #plt.show()
#
#     return image_dict

# def load_image(image_path: str):
#     image_np = load_image_into_numpy_array(image_path)
#
#     return image_np

# def detect_images(detect_fn, image_dict):
#     '''
#
#
#     :param detect_fn:
#     :param image_dict:
#
#     :return:
#     '''
#     elapsed = []
#     detection_dict = dict()
#
#     #print("Start detection")
#     for image_name, value in image_dict.items():
#         image_np, input_tensor = value
#         start_time = time.time()
#         detections = detect_fn(input_tensor)
#         end_time = time.time()
#         diff = end_time - start_time
#         elapsed.append(diff)
#         print("Inference time image {} : {}s".format(image_name, diff))
#
#         detections['detection_boxes'][0].numpy(),
#         detections['detection_classes'][0].numpy().astype(np.int32),
#         detections['detection_scores'][0].numpy(),
#
#         detection_dict[image_name] = (image_np, detections['detection_boxes'][0].numpy(),
#                                       detections['detection_classes'][0].numpy().astype(np.int32),
#                                       detections['detection_scores'][0].numpy())
#
#     mean_elapsed = sum(elapsed) / float(len(elapsed))
#     #print('Mean elapsed time: ' + str(mean_elapsed) + 's/image')
#
#     return detection_dict

def plot_two_images(image, image2, title1="Image1", title2="Image2"):
    ax = plt.subplot(121)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title(title1, fontsize=40)
    plt.axis('off')
    plt.imshow(image)


    ax = plt.subplot(122)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title(title2, fontsize=40)
    plt.imshow(image2)

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()

def plot_three_images(image, image2, image3, title1="Image1", title2="Image2", title3="Image3"):
    ax = plt.subplot(131)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title(title1, fontsize=20)
    plt.imshow(image)
    plt.axis('off')

    ax = plt.subplot(132)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title(title2, fontsize=20)
    plt.imshow(image2)

    ax = plt.subplot(133)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title(title3, fontsize=20)
    plt.imshow(image3)

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()

# def visualize_image_with_boundingbox(annotation_dir, category_index, image_name, image_path):
#     image_np = load_image(image_path)
#     filter = []
#     filter.append(image_name)
#     annotation = xml_to_csv(annotation_dir, filter)
#     print("Annotation: ", annotation)
#     boxes, classes, scores = extract_info_from_annotations(annotation, category_index)
#     fig1 = visualize_image(image_name, image_np, scores, boxes, classes, category_index)
#
#     return fig1


# def extract_info_from_annotations(annotation, category_index):
#
#     boxes = np.zeros([annotation.shape[0], 4])
#     classes = np.zeros([annotation.shape[0]]).astype('int')
#     scores = np.zeros([annotation.shape[0]])
#
#     for i in range(annotation.shape[0]):
#         boxes[i][0] = annotation['ymin'][i] / annotation['height'][i]
#         boxes[i][1] = annotation['xmin'][i] / annotation['width'][i]
#         boxes[i][2] = annotation['ymax'][i] / annotation['height'][i]
#         boxes[i][3] = annotation['xmax'][i] / annotation['width'][i]
#             #[annotation['ymin'][i] / annotation['height'][i],
#             # annotation['xmin'][i] / annotation['width'][i],
#             # annotation['ymax'][i] / annotation['height'][i],
#             ## annotation['xmax'][i] / annotation['width'][i]]).reshape(i, 4)
#         print("Boxes: ", boxes)
#
#         class_index = [category_index[j + 1].get('id') for j in range(len(category_index)) if
#                        category_index[j + 1].get('name') == annotation['class'][i]][0]
#
#         classes[i] = class_index
#         print("Class index: ", class_index)
#
#         if 'scores' in annotation.columns:
#             scores[i] = annotation['scores'][i]
#         else:
#             scores[i] = 1.0
#
#
#     return boxes, classes, scores

def visualize_images(image_path1, image_path2, image_path3,
                     annotation_dir1, annotation_dir2, annotation_dir3,
                     title1, title2, title3,
                     labelmap, output_dir, use_three_images, color_gt):
    '''Main method'''

    # Get all paths
    #image_path1 = args.image_path1
    if args.image_path2 == "None":
        image_path2 = image_path1
    #else:
    #    image_path2 = image_path2

    if image_path3 == "None":
        image_path3 = image_path1
    #else:
    #    image_path3 = image_path3

    #annotation_dir1 = args.annotation_dir1
    if annotation_dir2 == "None":
        annotation_dir2 = annotation_dir1
    #else:
    #annotation_dir2 = annotation_dir2
    if annotation_dir3 == "None":
        annotation_dir3 = annotation_dir1
    #else:
    #annotation_dir3 = annotation_dir3

    #output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created result directory: ", output_dir)

    # Load images and image names
    image_filename1 = os.path.basename(image_path1)
    image_name1 = os.path.splitext(image_filename1)[0]
    image_filename2 = os.path.basename(image_path2)
    image_name2 = os.path.splitext(image_filename2)[0]
    if use_three_images:
        image_filename3 = os.path.basename(image_path3)
        image_name3 = os.path.splitext(image_filename3)[0]

    # Load label path
    category_index = load_labelmap(os.path.abspath(labelmap))

    # Generate the images with bounding boxes
    image1 = bbox.visualize_image_with_boundingbox(annotation_dir1, category_index, image_name1, image_path1, color_gt=color_gt)
    image2 = bbox.visualize_image_with_boundingbox(annotation_dir2, category_index, image_name2, image_path2, color_gt=color_gt)
    if use_three_images:
        image3 = bbox.visualize_image_with_boundingbox(annotation_dir3, category_index, image_name3, image_path3,
                                                       color_gt=color_gt)
        fig = plot_three_images(image1, image2, image3, title1, title2, title3)
        out_name = "bbox_" + image_name1 + "_" + image_name2 + "_" + image_name3 + ".jpg"
    else:
        fig = plot_two_images(image1, image2, title1, title2)
        out_name = "bbox_" + image_name1 + "_" + image_name2 + ".jpg"

    plt.savefig(os.path.join(output_dir, out_name))
    print("Saved output image to ", os.path.join(output_dir, out_name))

    #plt.show()

if __name__ == "__main__":
    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")
    visualize_images(args.image_path1, args.image_path2, args.image_path3,
                     args.annotation_dir1, args.annotation_dir2, args.annotation_dir3,
                     args.title1, args.title2, args.title3,
                     args.labelmap, args.output_dir, args.use_three_images, args.color_gt)

    print("=== Program end ===")
