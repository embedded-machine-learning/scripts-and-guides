#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert yolo to Coco.

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
# Source: https://towardsai.net/p/deep-learning/cvml-annotation%e2%80%8a-%e2%80%8awhat-it-is-and-how-to-convert-it

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
import sys
import ntpath
import convert_cvml_to_coco
from PIL import Image
import os

import os
import numpy as np
import dicttoxml
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from tqdm import tqdm
import shutil
import json
import pandas as pd

# Own modules

__author__ = 'Julian Westra'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Alexander Wendt', 'Rohit Verma']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'


parser = argparse.ArgumentParser(description='Convert CVML to Coco')
parser.add_argument("-ad", '--annotation_dir',
                    default=None,
                    help='Annotation directory with xml files of the same format as image files', required=False)
parser.add_argument("-af", '--annotation_file',
                    default="samples/annotations/yolo_ground_truth.txt",
                    help='Annotation file.', required=False)
parser.add_argument("-if", '--image_file',
                    default=None,
                    help='Image file path for a certain image file', required=False)
parser.add_argument("-id", '--image_dir',
                    default="samples/yolo_images",
                    help='Image file directory', required=False)
parser.add_argument("-label", '--label_name',
                    default="pedestrian",
                    help='Label of the set in a binary set. Default is pedestrian.', required=False)
parser.add_argument("-pre", '--image_name_prefix',
                    default="",
                    help='Name prefix for images', required=False)
parser.add_argument("-del", '--delete_csv', action='store_true', default=False, help='Delete csv', required=False)

args = parser.parse_args()

def generate_filename_from_id(id: str, image_name_prefix: str):
    if not image_name_prefix:
        return "{:04d}".format(int(id)) + ".jpg"
    else:
        return image_name_prefix + "_" + "{:04d}".format(int(id)) + ".jpg"

def txt_to_csv_convert(annotation_file: str, image_dict: dict, label_name: str, image_name_prefix: str) -> pd.DataFrame:

    label = label_name


    f = open(annotation_file, "r").readlines()

    # Load first line
    line = f[0]
    txt_line_values = line.split(",")
    id = txt_line_values[0]
    file_name = generate_filename_from_id(id, image_name_prefix)
    temp_file_name = file_name #"1"

    if temp_file_name not in image_dict:
        raise Exception("First image of file does not exist ", temp_file_name)

    temp_csv_line = ""

    combined = []
    for line in f:
        txt_line_values = line.split(",")
        id=txt_line_values[0]
        line_file_name = generate_filename_from_id(id, image_name_prefix)

        if line_file_name not in image_dict:
            raise Exception(" Warning: Image does not exist ", line_file_name)

        if temp_file_name != line_file_name:
            combined.append([temp_file_name, temp_csv_line.strip()])
            temp_csv_line = ""

        temp_file_name = line_file_name

        image_width = image_dict[file_name]['width']
        image_height = image_dict[file_name]['height']
        print("Processing ", temp_file_name)

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)

        #YOLO
        #x_center = float(txt_line_values[2])
        #y_center = float(txt_line_values[3])
        #width = float(txt_line_values[4])
        #height = float(txt_line_values[5])

        #CUSTOM
        x_upper = float(txt_line_values[2])
        y_upper = float(txt_line_values[3])
        width = float(txt_line_values[4])
        height = float(txt_line_values[5])

        #int_x_center = int(image_width * x_center)
        #int_y_center = int(image_height * y_center)
        #int_width = int(image_width * width)
        #int_height = int(image_height * height)

        #min_x = int_x_center - int_width / 2
        #min_y = int_y_center - int_height / 2
        #width = int_width
        #height = int_height

        #temp_csv_line += (str(x_upper) + " " + str(y_upper) + " " + str(width) + " " + str(height) + " " + label + " ")


        x1 = txt_line_values[2]
        y2 = int(np.round(max(float(txt_line_values[3]) + float(txt_line_values[5]), 0)))
        x2 = int(np.round(min(float(txt_line_values[2]) + float(txt_line_values[4]), image_width)))
        y1 = txt_line_values[3]
        temp_csv_line += (str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + label + " ")

    combined.append([temp_file_name, temp_csv_line.strip()])
    dataframe = pd.DataFrame(combined, columns=["ID", "Label"])

    return dataframe
    #dataframe.to_csv(filename + ".csv", index=False)

def cvml_to_coco(annotation_file, annotation_directory, image_file, image_dir, label_name, image_name_prefix, delete_csv=False):
    '''


    '''

    if image_file is not None:
        print("Convert one image file with specified name.")
        raise NotImplemented("Function not implemented yet")

    elif image_dir is not None:
        print("Convert images in a folder with certain structure.")
        #Get all images into a list
        images = [f for f in os.listdir(image_dir)
                  if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

        image_dict = dict()
        for image_name in images:
            image_path = os.path.join(image_dir, image_name)
            # Compute the width and height of a frame
            example_frame_image = Image.open(image_path)
            frame_width, frame_height = example_frame_image.size
            image_dict[image_name] = dict()
            image_dict[image_name]['height'] = frame_height
            image_dict[image_name]['width'] = frame_width


    #Cover cases
    if annotation_file is not None:
        print("Single fat file used with non conform file name syntax. Use this file as a base")
        annotation_file_name = annotation_file
    elif annotation_directory is not None:
        print("Multiple files used with the syntax of the image files")
        annotation_file_name = ntpath.basename(annotation_directory).split(".")[0]
        raise NotImplemented("Function not implemented yet")
    else:
        raise Exception("Unallowed annotion combination")




    # Script takes 2 arguments: annotation file path and example frame path
    #annotation_file_path = sys.argv[1]
    #example_frame_path = sys.argv[2]

    # Compute the filename from file path
    #annotation_file_name = ntpath.basename(annotation_file_path).split(".")[0]

    # Compute the width and height of a frame
    #example_frame_image = Image.open(example_frame_path)
    #frame_width, frame_height = example_frame_image.size

    # Convert .txt to .csv file
    #txt_to_csv.convert(annotation_file_path, frame_width, frame_height, annotation_file_name)
    dataframe = txt_to_csv_convert(annotation_file, image_dict, label_name, image_name_prefix)

    dataframe.to_csv(annotation_file + ".csv", index=False)  # dataframe.to_csv(filename + ".csv", index=False)

    # Convert .csv to .json file
    #txt_to_csv_convert(annotation_file_path, frame_width, frame_height)
    convert_cvml_to_coco.convert_csv_coco(annotation_file + ".csv", image_dict)

    print("Saved csv: ", annotation_file + ".csv")

    # Remove interim csv file
    if delete_csv:
        os.remove(annotation_file + ".csv")
        print("Deleted csv.")

if __name__ == "__main__":
    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Script takes 2 arguments: Annotation file path, example frame path
    #annotation_file_directory = sys.argv[1]
    #example_frame_path = sys.argv[2]

    cvml_to_coco(args.annotation_file, args.annotation_dir, args.image_file,
                 args.image_dir, args.label_name, args.image_name_prefix, args.delete_csv)


    print("=== Program end ===")