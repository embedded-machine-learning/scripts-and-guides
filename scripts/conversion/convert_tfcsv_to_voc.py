#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert TF2 CSV format to Pascal VOC

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
# Source: https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd
# Thanks for the Inspiration

"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
import json
import os

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
from object_detection.utils import label_map_util

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Shubham Gupta, https://gist.github.com/goodhamgupta']
__license__ = 'ISC'
__version__ = '0.5.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Convert Coco to VOC')
#parser.add_argument("-ad", '--annotation_dir',
#                    default=None,
#                    help='Annotation directory with xml files of the same format as image files', required=False)
parser.add_argument("-af", '--annotation_file',
                    default="samples/annotations/tf2csv_format_detections.csv",
                    help='Annotation file.', required=False)
parser.add_argument("-o", '--output_dir',
                    default="samples/annotations/xml_voc",
                    help='Annotation file.', required=False)
parser.add_argument("-lm", '--labelmap_file',
                    default="samples/annotations/pedestrian_label_map.pbtxt",
                    help='Labelmap file.', required=False)
#parser.add_argument("-b", '--bbox_offset', type=int,
#                    default=0,
#                    help='BOX_OFFSET: Switch between 0-based and 1-based bbox. The COCO dataset is in 0-based format, '
#                         'while the VOC dataset is 1-based. To keep 0-based, set it to 0. To convert to 1-based, '
#                         'set it to 1.', required=False)
#parser.add_argument("-id", '--image_dir',
#                    default="samples/cvml_images",
#                    help='Image file directory for writing the trainval.txt file', required=False)
#parser.add_argument("-label", '--label_name',
#                    default="pedestrian",
#                    help='Label of the set in a binary set. Default is pedestrian.', required=False)
#parser.add_argument("-del", '--delete_csv', default="True", help='Delete csv', required=False)

args = parser.parse_args()

# Script to convert yolo annotations to voc format from
# https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd

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


#ANNOTATIONS_DIR_PREFIX = "annotations"

#DESTINATION_DIR = "converted_labels"

#CLASS_MAPPING = {
#    '0': 'name'
#    # Add your remaining classes here.
#}


def create_root(image_filename, width, height):
    '''
    Create pascal root element

    '''
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = image_filename
    ET.SubElement(root, "folder").text = ""
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        ET.SubElement(obj, "score").text = str(voc_label[5])
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

def prettify(elem):
    '''
    Return a pretty-printed XML string for the Element.

    '''

    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def create_file(image_filename, width, height, voc_labels, output_dir):
    root = create_root(image_filename, width, height)
    root = create_object_annotation(root, voc_labels)

    with open(os.path.join(output_dir, os.path.splitext(image_filename)[0] + ".xml"), "w") as f:
        f.write(prettify(root))


def generate_voc(filecontent, label_map_dict, output_dir):
    #file_prefix = file_path.split(".txt")[0]
    #image_file_name = "{}.jpg".format(file_prefix)
    #img = Image.open("{}/{}".format("images", image_file_name))
    #w, h = img.size
    #with open(file_path, 'r') as file:
    #    lines = file.readlines()
        #voc_labels = []
    #    for line in lines:
    filename = filecontent.iloc[0].name

    width = int(filecontent.iloc[0]['width'])
    height = int(filecontent.iloc[0]['height'])

    voc_labels = []

    for index, row in filecontent.iterrows():
        voc = []
        #        line = line.strip()
        #        data = line.split()
        voc.append(label_map_dict.get(row['class']))
        #bbox_width = float(data[3]) * w
        #bbox_height = float(data[4]) * h
        #center_x = float(data[1]) * w
        #center_y = float(data[2]) * h



        voc.append(int(np.round(row['xmin'] * width))) #center_x - (bbox_width / 2))
        voc.append(int(np.round(row['ymin'] * height))) #center_y - (bbox_height / 2))
        voc.append(int(np.round(row['xmax'] * width))) #center_x + (bbox_width / 2))
        voc.append(int(np.round(row['ymax'] * height))) #center_y + (bbox_height / 2))

        voc.append(row['score'])

        voc_labels.append(voc)

    create_file(filename, width, height, voc_labels, output_dir)
    print("Processing complete for file: {}".format(filename))


def convert_csv_to_voc(annotation_file, output_dir, labelmap_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_map = label_map_util.load_labelmap(labelmap_file)
    label_map_dict = label_map_util.get_label_map_dict(label_map)
    label_map_dict_inverse = {v: k for k, v in label_map_dict.items()}

    #for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
    #    if filename.endswith('txt'):
    ann_df = pd.read_csv(annotation_file, sep=';')
    ann_df.set_index('filename', inplace=True)
    group = ann_df.groupby(ann_df.index)
    for key in group.groups.keys():
        print("Process ",key)
        df = ann_df.loc[key]
        if isinstance(df, pd.Series):
            df = pd.DataFrame([ann_df.loc[key]])

        generate_voc(df, label_map_dict_inverse, output_dir)

    print("Processing finished")


    #read_file(ann_df)
    #else:
    #        print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    convert_csv_to_voc(args.annotation_file, args.output_dir, args.labelmap_file)

    print("=== Program end ===")




# import cv2
# from lxml.etree import Element, SubElement, tostring
# from xml.dom.minidom import parseString
# import os
#
#
# #Source: https://www.programmersought.com/article/2053478686/
#
# def save_xml(image_name, bbox, save_dir='./VOC2007/Annotations', width=1609, height=500, channel=3):
#     node_root = Element('annotation')
#
#     node_folder = SubElement(node_root, 'folder')
#     node_folder.text = 'JPEGImages'
#
#     node_filename = SubElement(node_root, 'filename')
#     node_filename.text = image_name
#
#     node_size = SubElement(node_root, 'size')
#     node_width = SubElement(node_size, 'width')
#     node_width.text = '%s' % width
#
#     node_height = SubElement(node_size, 'height')
#     node_height.text = '%s' % height
#
#     node_depth = SubElement(node_size, 'depth')
#     node_depth.text = '%s' % channel
#
#     for x, y, x1, y1 in bbox:
#         left, top, right, bottom = x, y, x1, y1
#         node_object = SubElement(node_root, 'object')
#         node_name = SubElement(node_object, 'name')
#         node_name.text = 'person'
#         node_difficult = SubElement(node_object, 'difficult')
#         node_difficult.text = '0'
#         node_bndbox = SubElement(node_object, 'bndbox')
#         node_xmin = SubElement(node_bndbox, 'xmin')
#         node_xmin.text = '%s' % left
#         node_ymin = SubElement(node_bndbox, 'ymin')
#         node_ymin.text = '%s' % top
#         node_xmax = SubElement(node_bndbox, 'xmax')
#         node_xmax.text = '%s' % right
#         node_ymax = SubElement(node_bndbox, 'ymax')
#         node_ymax.text = '%s' % bottom
#
#     xml = tostring(node_root, pretty_print=True)
#     dom = parseString(xml)
#
#     save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
#     with open(save_xml, 'wb') as f:
#         f.write(xml)
#
#     return
#
#
# def change2xml(label_dict={}):
#     for image in label_dict.keys():
#         image_name = os.path.split(image)[-1]
#         bbox = label_dict.get(image, [])
#         save_xml(image_name, bbox)
#     return
#
#
# import pandas as pd
# import numpy as np
#
# data = pd.read_table("path/train_labels.csv", sep=",")
# name_file = open('path/ImageSets/Main/train.txt', 'r')
# name_file = name_file.readlines()
# for name in name_file:
#     img = cv2.imread('path/JPEGImages/' + name[:-1] + '.jpg')
#     height, width = img.shape[:2]
#     name = name[:-1] + '.jpg'
#     xx = np.array(data[data['ID'] == name][' Detection'])
#     bbox = []
#     for i in range(xx.shape[0]):
#         bbox.append(xx[i].split(' '))
#     save_xml(image_name=name, bbox=bbox, save_dir='path/Annotations', width=width, height=height, channel=3)