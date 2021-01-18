#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert CVML to Coco.

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
                    default="samples/annotations/cvml_xml/cvml_Milan-PETS09-S2L1.xml",
                    help='Annotation file.', required=False)
parser.add_argument("-if", '--image_file',
                    default=None,
                    help='Image file path for a certain image file', required=False)
parser.add_argument("-id", '--image_dir',
                    default="samples/cvml_images",
                    help='Image file directory', required=False)
parser.add_argument("-label", '--label_name',
                    default="pedestrian",
                    help='Label of the set in a binary set. Default is pedestrian.', required=False)
parser.add_argument("-pre", '--image_name_prefix',
                    default="frame_",
                    help='Name prefix for images', required=False)
parser.add_argument("-del", '--delete_csv', action='store_true', default=False, help='Delete csv', required=False)

args = parser.parse_args()


def convert_cvml_csv(annotation_file: str, image_dict: dict, label_name: str, image_name_prefix: str) -> pd.DataFrame:
    '''
    Convert from CVML to csv

    '''
    # Loading the annotation files

    #annotation_file = annotation_directory

    # Load annotation file
    file = open(annotation_file, "r")
    my_xml_file = file.read()
    annotation_dictionary = dict(dict(xmltodict.parse(my_xml_file))["dataset"])

    # Iterating over different frames, finding the bounding box details and write it in the pandas dataframe
    warn_message = []
    combined = []
    count = 0
    for frame in tqdm(annotation_dictionary["frame"]):
        #############################################################
        # Adapt the file name to the used
        #############################################################
        file_name = image_name_prefix + "{:04d}".format(count) + ".jpg"

        if file_name in image_dict:
            #Adapt height and width for each file individually
            image_width = image_dict[file_name]['width']
            image_height = image_dict[file_name]['height']
            print("Processing ", file_name)

            label_string = ""
            if type(frame["objectlist"]) == collections.OrderedDict:
                if type(frame["objectlist"]["object"]) == list:
                    for j, i in enumerate(frame["objectlist"]["object"]):
                        x1 = int(np.round(max(float(i["box"]["@xc"]) - float(i["box"]["@w"]) / 2, 0)))
                        y1 = int(np.round(max(float(i["box"]["@yc"]) - float(i["box"]["@h"]) / 2, 0)))
                        x2 = int(np.round(min(float(i["box"]["@xc"]) + float(i["box"]["@w"]) / 2, image_width)))
                        y2 = int(np.round(min(float(i["box"]["@yc"]) + float(i["box"]["@h"]) / 2, image_height)))
                        label = label_name
                        label_string += (
                                str(x1)
                                + " "
                                + str(y1)
                                + " "
                                + str(x2)
                                + " "
                                + str(y2)
                                + " "
                                + label
                                + " "
                        )

                else:
                    x1 = int(np.round(max(0, float(frame["objectlist"]["object"]["box"]["@xc"]) - float(frame["objectlist"]["object"]["box"]["@w"])/2,)))
                    y1 = int(np.round(max(
                        0,
                        float(frame["objectlist"]["object"]["box"]["@yc"])
                        - float(frame["objectlist"]["object"]["box"]["@h"]) / 2,
                    )))
                    x2 = int(np.round(min(
                        image_width,
                        float(frame["objectlist"]["object"]["box"]["@xc"])
                        + float(frame["objectlist"]["object"]["box"]["@w"]) / 2,
                    )))
                    y2 = int(np.round(min(
                        image_height,
                        float(frame["objectlist"]["object"]["box"]["@yc"])
                        + float(frame["objectlist"]["object"]["box"]["@h"]) / 2,
                    )))
                    label = label_name
                    label_string += (
                            str(x1)
                            + " "
                            + str(y1)
                            + " "
                            + str(x2)
                            + " "
                            + str(y2)
                            + " "
                            + label
                            + " "
                    )

            combined.append([file_name, label_string.strip()])
        else:
            warn_message.append(str(file_name))

        count += 1

    if len(warn_message)>0:
        warnings.warn("Warning: No files with the following names found in the list of images. Image does not exist. "
                                               "Adapt the file name prefix and number counter in the script "
                                               "at this place.")
        print(warn_message)

    dataframe = pd.DataFrame(combined, columns=["ID", "Label"])
    #dataframe.to_csv(annotation_file + ".csv", index=False) #dataframe.to_csv(filename + ".csv", index=False)

    return dataframe

    #return annotation_file + ".csv"


def convert_csv_coco(transformed_csv_filename: str, image_dict: dict)-> str:
    '''
    Convert from a CSV file to Coco


    '''

    # Loading files and setting directories

    #root = "./"
    annotation_csv_file = transformed_csv_filename #transformed_csv_filename + ".csv"

    annotations_path = os.path.join(os.path.dirname(transformed_csv_filename), "annotations")
    annotations_filename = os.path.basename(transformed_csv_filename).split('.')[0]

    if not os.path.isdir(annotations_path):
        os.makedirs(annotations_path)

    input_annotation_path = annotation_csv_file

    output_annotation_folder = annotations_path

    output_annotation_file = os.path.join(output_annotation_folder, annotations_filename + "_coco.json")
    output_classes_file = output_annotation_folder + "/classes.txt"

    dataframe = pd.read_csv(input_annotation_path)
    columns = dataframe.columns

    delimiter = " "

    # creating text file that contains all label classes

    list_dictionary = []
    annotation_list = []

    for i in range(len(dataframe)):
        img_name = dataframe[columns[0]][i]
        labels = dataframe[columns[1]][i]
        tmp = str(labels).split(delimiter)
        for j in range(len(tmp) // 5):
            label = tmp[j * 5 + 4]
            if label not in annotation_list:
                annotation_list.append(label)
        annotation_list = sorted(annotation_list)

    for i in tqdm(range(len(annotation_list))):
        tmp = {}
        tmp["supercategory"] = "master"
        tmp["id"] = i
        tmp["name"] = annotation_list[i]
        list_dictionary.append(tmp)

    annotation_file = open(output_classes_file, "w")

    for i in range(len(annotation_list)):
        annotation_file.write(annotation_list[i] + "\n")
    annotation_file.close()

    # converting the csv file to COCO format

    coco_data = {}
    coco_data["type"] = "instances"
    coco_data["images"] = []
    coco_data["annotations"] = []
    coco_data["categories"] = list_dictionary
    image_id = 0
    annotation_id = 0

    for i in tqdm(range(len(dataframe))):
        image_name = dataframe[columns[0]][i]
        labels = dataframe[columns[1]][i]
        tmp = str(labels).split(delimiter)
        images_tmp = {}
        images_tmp["filename"] = image_name

        #Use real image height and width
        images_tmp["height"] = image_dict[image_name]["height"]
        images_tmp["width"] = image_dict[image_name]["width"]

        #images_tmp["height"] = image_height
        #images_tmp["width"] = image_width
        images_tmp["id"] = image_id
        coco_data["images"].append(images_tmp)
        for j in range(len(tmp) // 5):
            #x1 = float(tmp[j * 5 + 0])
            #y1 = float(tmp[j * 5 + 1])
            #x2 = float(tmp[j * 5 + 2])
            #y2 = float(tmp[j * 5 + 3])
            x1 = int(tmp[j * 5 + 0])
            y1 = int(tmp[j * 5 + 1])
            x2 = int(tmp[j * 5 + 2])
            y2 = int(tmp[j * 5 + 3])
            label = tmp[j * 5 + 4]
            annotations_tmp = {}
            annotations_tmp["id"] = annotation_id
            annotation_id += 1
            annotations_tmp["image_id"] = image_id
            annotations_tmp["segmentation"] = []
            annotations_tmp["ignore"] = 0
            annotations_tmp["area"] = (x2 - x1) * (y2 - y1)
            annotations_tmp["iscrowd"] = 0
            annotations_tmp["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            annotations_tmp["category_id"] = annotation_list.index(label)
            coco_data["annotations"].append(annotations_tmp)

        image_id += 1

    outfile = open(output_annotation_file, "w")
    json_string = json.dumps(coco_data, indent=4)
    outfile.write(json_string)
    outfile.close()
    print("Saved coco annotations in ", output_annotation_file)

    return output_annotation_file

def cvml_to_coco(annotation_file, annotation_directory, image_file, image_dir, label_name,
                 image_name_prefix, delete_csv=False):
    '''
    Convert CVML to Coco through a csv intermediate format

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





    # Compute the filename
    #annotation_file_name = ntpath.basename(annotation_file_directory).split(".")[0]

    # Convert to csv
    #xml_to_csv.convert(
    #    annotation_file_directory, frame_width, frame_height, annotation_file_name
    #)
    #convert_cvml_csv(annotation_file_directory, frame_width, frame_height, annotation_file_name)
    dataframe = convert_cvml_csv(annotation_file, image_dict, label_name, image_name_prefix)

    dataframe.to_csv(annotation_file + ".csv", index=False)  # dataframe.to_csv(filename + ".csv", index=False)

    # Convert to json
    #csv_to_json.convert(annotation_file_name, frame_width, frame_height)
    convert_csv_coco(annotation_file + ".csv", image_dict)
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