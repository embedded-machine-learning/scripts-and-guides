#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert Yolo to Tensorflow CSV file format for detections. This converter is used to get yolo detections into the
same format as Tensorflow detections.

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
                    help='Image file directory to get the image size from the corresponding image', required=False)
parser.add_argument("-out", '--output',
                    default="./detections.csv",
                    help='Output file path for the detections csv.', required=False)
#parser.add_argument("-cl", '--class_file',
#                    default="./annotations/labels.txt",
#                    help='File with class labels', required=False)
#parser.add_argument("--create_empty_images", action='store_true', default=False,
#                    help="Generates xmls also for images without any found objects, i.e. empty annotations. It is useful to prevent overfitting.")

args = parser.parse_args()
print(args)
