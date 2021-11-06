#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert TF2 CSV format to PyCoco detections for evaluation with PyCocoTools

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
#from __future__ import print_function

# Built-in/Generic Imports
import json
import os
import warnings

# Libs
import lxml
from tqdm import tqdm
from xmltodict import unparse
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import bs4
from PIL import Image
import pandas as pd
import numpy as np

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.5.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Convert Coco to VOC')
parser.add_argument("-af", '--annotation_file',
                    default="samples/annotations/tf2csv_format_detections.csv",
                    help='Annotation file.', required=False)
parser.add_argument("-o", '--output_file',
                    default="samples/annotations/coco_detections.json",
                    help='Annotation file.', required=False)
args = parser.parse_args()

# Script to convert yolo annotations to voc format from
#

# Sample format
# [
#   {
#     "image_id": 42,
#     "category_id": 18,
#     "bbox": [
#       258.15,
#       41.29,
#       348.26,
#       243.78
#     ],
#     "score": 0.236
#   },
#   {
#     "image_id": 73,
#     "category_id": 11,
#     "bbox": [
#       61,
#       22.75,
#       504,
#       609.67
#     ],
#     "score": 0.318
#   }
# ]

def single_detection_to_coco_detection_dict(image_id: str, category_id: int, bbox: list, score: float) -> dict:
    '''
    Create a dictionary from inputs for a coco detection

    '''
    detection = {}
    detection['image_id'] = image_id
    detection['category_id'] = category_id
    detection['bbox'] = bbox
    detection['score'] = score

    return detection

def generate_detections(annotation_file, output_file):
    '''

    '''

    ann_df = pd.read_csv(annotation_file, sep=';')
    ann_df.set_index('filename', inplace=True)

    detected_objects = []
    for index, row in ann_df.iterrows():
        print("Process ", row.name)
        image_id = os.path.splitext(row.name)[-2]
        width = row['width']
        height = row['height']
        category_id = int(row['class'])

        # Round first at 2 decimals
        xmin = float(row['xmin'] * width) - 1
        #xmin = int(np.round(row['xmin'] * width)) - 1
        if xmin<0:
            warnings.warn("xmin < 0. Setting xmin=0")
            xmin=0
        ymin = float(row['ymin'] * height) - 1
        #ymin = int(np.round(row['ymin'] * height)) - 1
        if ymin<0:
            warnings.warn("ymin < 0. Setting ymin=0")
            ymin=0
        xmax = float(row['xmax'] * width)
        #xmax = int(np.round(row['xmax'] * width))
        if xmax>width:
            warnings.warn("xmax {} > {}. Setting xmax={}".format(xmax, width, width))
            xmax=width
        ymax = float(row['ymax'] * height)
        #ymax = int(np.round(row['ymax'] * height))
        if ymax>height:
            warnings.warn("ymax {} > {}. Setting ymax={}".format(ymax, height, height))
            ymax=height
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin
        bbox = [xmin, ymin, o_width, o_height]
        score = float(row['score'])

        object_dict = single_detection_to_coco_detection_dict(image_id, category_id, bbox, score)
        detected_objects.append(object_dict)

    # Save as json
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        print("Created results dir ", os.path.dirname(output_file))

    with open(output_file, 'w') as outfile:
        json.dump(detected_objects, outfile, indent=4)

    print("Saved coco detections to ", output_file)


if __name__ == "__main__":
    generate_detections(args.annotation_file, args.output_file)

    print("=== Program end ===")