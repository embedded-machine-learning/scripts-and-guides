#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate validation data and coco data for object detection with the COCO evaluation metrics.

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
# Source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

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
from pycocotools.coco import COCO
from tqdm import tqdm
from xmltodict import unparse
import xml.etree.ElementTree as ET
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pycocotools.cocoeval import COCOeval

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
parser.add_argument("-i", '--gt_path', default='gt.json',
                    help='Coco ground truth path', required=False)
parser.add_argument("-i", '--det_path', default='det.json',
                    help='Coco detections path', required=False)
args = parser.parse_args()


def evaluate_inference(coco_gt_file, coco_det_file):
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here 1: Bounding box
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    print('Running evaluation for *%s* results.' % (annType))

    # initialize COCO ground truth api
    #dataDir = '../'
    #dataType = 'val2014'
    #annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    cocoGt = COCO(coco_gt_file)

    # initialize COCO detections api
    #resFile = '%s/results/%s_%s_fake%s100_results.json'
    #resFile = resFile % (dataDir, prefix, dataType, annType)
    cocoDt = cocoGt.loadRes(coco_det_file)

    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()

    print(cocoEval.summarize())









if __name__ == "__main__":

    evaluate_inference(args.gt_path, args.det_path)

    print("=== Program end ===")