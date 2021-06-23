#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert VOC data to custom text format
Data annotation file format:
One row for one image in annotation file;
Row format: image_file_path box1 box2 ... boxN;
Box format: x_min,y_min,x_max,y_max,class_id (no space).
Example: path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3

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

The following code has been copied and adapted from
#====================================================================================
https://github.com/david8862/keras-YOLOv3-model-set

MIT License
Copyright (c) 2019 david8862

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
#=====================================================================================


"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
import os
import glob
import os, argparse
import warnings

import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import OrderedDict

# Libs
from tqdm import tqdm
from xmltodict import unparse
import argparse

from PIL import Image
import imagehash

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['david8862']
__license__ = 'ISC'
__version__ = '0.5.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='convert PascalVOC dataset annotation to txt annotation file')
parser.add_argument('--annotations_dir', type=str, help='Path to Pascal VOC XML folder', default='./annotations/xmls')
parser.add_argument('--image_dir', type=str, help='Path to image folder', default='./images/train')
parser.add_argument('--subset_textfile_path', type=str, help='If the subset file path is set, then the image ids are loaded from here. If it is not set,'
                                                             'then all pascal voc files xml files are used as an input.', default=None)
parser.add_argument('--year', type=str, help='subset path of year (2007/2012), default will cover both', default=None)
parser.add_argument('--set', type=str, help='convert data set, default will cover train, val and test', default=None)
parser.add_argument('--output_path', type=str,  help='output file path for generated annotation txt files', default='./annotations/yolo/train.txt')
parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions', default=None)
parser.add_argument('--include_difficult', action="store_true", help='to include difficult object', default=False)
parser.add_argument('--include_no_obj', action="store_true", help='to include no object image', default=False)
args = parser.parse_args()
print(args)

#sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#class_count = {}

def convert_annotation(annotations_dir, image_id, list_file, include_difficult, classes, class_count):
    xml_file = open(os.path.join(annotations_dir, image_id + ".xml"), encoding='utf-8')
    tree=ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        class_id = class_count[class_name] #classes.index(class_name)

        # parse box coordinate to (xmin,ymin,xmax,ymax) format
        xml_box = obj.find('bndbox')
        box = (int(float(xml_box.find('xmin').text)), int(float(xml_box.find('ymin').text)), int(float(xml_box.find('xmax').text)), int(float(xml_box.find('ymax').text)))
        # write box info to txt
        list_file.write(" " + ",".join([str(item) for item in box]) + ',' + str(class_id))
        #class_count[class_name] = class_count[class_name] + 1


def has_object(annotations_dir, image_id, include_difficult, classes):
    '''
    check if an image annotation has valid object bbox info,
    return a boolean result
    '''
    try:
        xml_file = open(os.path.join(annotations_dir, image_id + ".xml"), encoding='utf-8')
    except:
        # bypass image if no annotation
        return False
    tree=ET.parse(xml_file)
    root = tree.getroot()
    count = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        count = count + 1
    return count != 0


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


def voc_to_yolo():

    #sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]

    # update class names
    #if args.classes_path:
    classes = get_classes(args.classes_path)
    #class_count = len(classes)

    # get real path for dataset
    dataset_realpath = os.path.realpath(args.annotations_dir)

    # create output path
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # get specific sets to convert
    #if args.year is not None:
    #    sets = [item for item in sets if item[0] == args.year]
    #if args.set is not None:
    #    sets = [item for item in sets if item[1] == args.set]

    # count class item number in each set
    class_count = OrderedDict([(item, i) for i, item in enumerate(classes)])

    #for year, image_set in sets:
    #Load all image ids from existing pascal voc files
    if args.subset_textfile_path:
        print("Load subset of train image ids from txt file: ", args.subset_textfile_path)
        image_ids = open(args.subset_textfile_path).read().strip().split()
        print("Got image ids: ", image_ids)
    else:
        print("Use all images in the image folder")

        #types = ("*.jpg")
        print("Only *.jpg will be loaded.")

        image_ids = list()
        #for tp in types:
        #    if tp != "":
        ids_from_images = [os.path.basename(fpath).split(".")[:-1][0] for fpath in glob.glob(os.path.join(args.image_dir, "*.jpg"))]
        image_ids.extend(ids_from_images)

    list_file = open(os.path.join(args.output_path), 'w')
    pbar = tqdm(total=len(image_ids), desc='Converting VOC')
    for image_id in image_ids:
        file_string = os.path.join(args.image_dir, image_id + ".jpg").replace('\\', '/')
        # check if the image file exists
        if not os.path.exists(file_string):
            file_string = os.path.join(args.image_dir, image_id + ".jpeg").replace('\\', '/')
        if not os.path.exists(file_string):
            raise ValueError('image file for id: {} not exists'.format(image_id))

        if has_object(args.annotations_dir, image_id, args.include_difficult, classes):
            list_file.write(file_string)
            convert_annotation(args.annotations_dir, image_id, list_file, args.include_difficult, classes, class_count)
            list_file.write('\n')
        elif args.include_no_obj:
            warnings.warn("No xml file for image {}".format(image_id))
            # include no object image. just write file path
            #list_file.write(file_string)
            #list_file.write('\n')
        pbar.update(1)
    pbar.close()
    list_file.close()
    # print out item number statistic
    print('\nDone for VOC. classes number statistic')
    print('Image number: %d' % (len(image_ids)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))

if __name__ == '__main__':

    voc_to_yolo()

    print("=== Program end ===")
