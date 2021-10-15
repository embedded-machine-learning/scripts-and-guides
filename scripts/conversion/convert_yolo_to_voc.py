#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert Yolo to Pascal VOC.

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
# Source: https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd

"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
import re
import time
import json
import re
import ntpath
import warnings

# Libs
import argparse
import numpy as np
import glob
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import matplotlib
from six import BytesIO
import pandas as pd
import tkinter
import argparse
import collections
import xmltodict
from PIL import Image
import numpy as np
import dicttoxml
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from tqdm import tqdm
import shutil
import os
from xml.dom import minidom
import xml.etree.cElementTree as ET
from PIL import Image

# Own modules

__author__ = 'Julian Westra'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Alexander Wendt', 'https://gist.github.com/goodhamgupta']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Convert Yolo to Pascal VOC')
parser.add_argument("-ad", '--annotation_dir',
                    default=None,
                    help='Annotation directory with txt files of yolo annotations of the same name format as image files',
                    required=False)
parser.add_argument("-id", '--image_dir',
                    default="images",
                    help='Image file directory', required=False)
parser.add_argument("-at", '--target_annotation_dir',
                    default="./annotations/xmls",
                    help='Target directory for xml files', required=False)
parser.add_argument("-cl", '--class_file',
                    default="./annotations/labels.txt",
                    help='File with class labels', required=False)
parser.add_argument("--create_empty_images", action='store_true', default=False,
                    help="Generates xmls also for images without any found objects, i.e. empty annotations. It is useful to prevent overfitting.")

args = parser.parse_args()
print(args)

# Script to convert yolo annotations to voc format

# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>


# ANNOTATIONS_DIR_PREFIX = "path to your yolo annotations"

# DESTINATION_DIR = "output path"

# CLASS_MAPPING = {
#    '1': 'Char'
# }


def formatter(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")


def create_root(file_prefix, width, height):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    #if len(voc_labels) == 0:
    #    obj = ET.SubElement(root, "object")
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(file_prefix, width, height, voc_labels, target_annotation_dir):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    with open("{}/{}.xml".format(target_annotation_dir, file_prefix), "w") as f:
        f.write(formatter(root))
        f.close()


def read_image_file(image_file_path, annotation_dir, target_annotation_dir, classes_dict, image_dir,
                    create_empty_images=True):
    image_file_name = os.path.basename(image_file_path)
    file_prefix = os.path.basename(image_file_path).split('.jpg')[0]
    annotation_file_name = "{}.txt".format(file_prefix)
    annotation_file_path = os.path.join(annotation_dir, annotation_file_name)

    # file_prefix = file_path.split(".txt")[0]
    # image_file_name = "{}.jpg".format(file_prefix)
    # img = Image.open("{}/{}".format("sites", image_file_name))

    img = Image.open(image_file_path)
    w, h = img.size

    if os.path.exists(annotation_file_path):
        print("Convert annotation {}".format(annotation_file_path))
        # with open(os.path.join(annotation_dir, )"labels/" + file_path, 'r') as file:
        with open(os.path.join(annotation_file_path), 'r') as file:
            lines = file.readlines()
            voc_labels = []
            for line in lines:
                voc = []
                line = line.strip()
                data = line.split()
                # voc.append(CLASS_MAPPING.get(data[0]))
                voc.append(classes_dict.get(data[0]))
                bbox_width = float(data[3]) * w
                bbox_height = float(data[4]) * h
                center_x = float(data[1]) * w
                center_y = float(data[2]) * h
                voc.append(round(center_x - (bbox_width / 2)))
                voc.append(round(center_y - (bbox_height / 2)))
                voc.append(round(center_x + (bbox_width / 2)))
                voc.append(round(center_y + (bbox_height / 2)))
                voc_labels.append(voc)
            create_file(file_prefix, w, h, voc_labels, target_annotation_dir)
    elif create_empty_images:
        print("Annotation does not exist {}. Create empty annotation".format(annotation_file_path))
        voc_labels = []
        create_file(file_prefix, w, h, voc_labels, target_annotation_dir)
    else:
        print("Annotation does not exist {}. Do nothing".format(annotation_file_path))


def read_file(file_path, annotation_dir, target_annotation_dir, classes_dict, image_dir):
    file_prefix = file_path.split(".txt")[0]
    image_file_name = "{}.jpg".format(file_prefix)
    # img = Image.open("{}/{}".format("sites", image_file_name))

    if os.path.exists(os.path.join(image_dir, image_file_name)):
        img = Image.open(os.path.join(image_dir, image_file_name))
        w, h = img.size
        # with open(os.path.join(annotation_dir, )"labels/" + file_path, 'r') as file:
        with open(os.path.join(annotation_dir, file_path), 'r') as file:
            lines = file.readlines()
            voc_labels = []
            for line in lines:
                voc = []
                line = line.strip()
                data = line.split()
                # voc.append(CLASS_MAPPING.get(data[0]))
                voc.append(classes_dict.get(data[0]))
                bbox_width = float(data[3]) * w
                bbox_height = float(data[4]) * h
                center_x = float(data[1]) * w
                center_y = float(data[2]) * h
                voc.append(round(center_x - (bbox_width / 2)))
                voc.append(round(center_y - (bbox_height / 2)))
                voc.append(round(center_x + (bbox_width / 2)))
                voc.append(round(center_y + (bbox_height / 2)))
                voc_labels.append(voc)
            create_file(file_prefix, w, h, voc_labels, target_annotation_dir)
    else:
        warnings.warn("Image does not exist {}".format(os.path.join(image_dir, image_file_name)))


def load_classes(class_file):
    d = {}
    i = 0
    with open(class_file) as f:
        for line in f:
            d[str(i)] = line.strip()
            i = i + 1

    return d


def start(annotation_dir, target_annotation_dir, class_file, image_dir, create_empty_images=True):
    os.makedirs(target_annotation_dir, exist_ok=True)

    # Load class file to list
    classes_dict = load_classes(class_file)
    print("Loaded classes", classes_dict)

    print("Processing jpg files in the image folder ", image_dir)

    for image_path in glob.glob(image_dir + '/*.jpg'):
        print("Process image", image_path)
        read_image_file(image_path, annotation_dir, target_annotation_dir, classes_dict, image_dir, create_empty_images)

    # for filename in os.listdir(annotation_dir):
    #    if filename.endswith('txt'):
    #        print(filename)
    #        read_file(filename, annotation_dir, target_annotation_dir, classes_dict, image_dir)
    #    else:
    #        print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    start(args.annotation_dir, args.target_annotation_dir, args.class_file, args.image_dir, args.create_empty_images)

    print("=== Program end ===")
