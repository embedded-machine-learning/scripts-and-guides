#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Select two images with their bounding boxes in PASCAL VOC XML format and visualize then within one image.
OpenCV has been used to plot bounding boxes

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

# The following script uses method fragments from the github repository Pedestrian-Detection
https://github.com/thatbrguy/Pedestrian-Detection

"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
import time

# Libs
import tkinter
import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

#If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(description='Object Detector Visualizer')

parser.add_argument("--image_path1", type=str, default="images/1.jpg")
parser.add_argument("--image_path2", type=str, default="images/1000.jpg")
parser.add_argument("--annotation_dir1", type=str, default="annotations/xmls")
parser.add_argument("--annotation_dir2", type=str, default="annotations/xmls")
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--line_thickness", type=int, default=4)
#parser.add_argument("--imageset_dir", type=str, default="train.txt")

args = parser.parse_args()

class Entity():
    def __init__(self, name, xmin, xmax, ymin, ymax, difficult, truncated):
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.difficult = difficult
        self.truncated = truncated


class Data():
    def __init__(self, image_dir, annotation_dir, image_name):
        self.image_name = image_name
        self.image_path = os.path.join(image_dir, image_name + ".jpg")
        self.annotation_path = os.path.join(annotation_dir, image_name + ".xml")
        self.annotations = self.load_masks()

    def load_masks(self):
        annotations = []
        xml_content = open(self.annotation_path).read()
        bs = BeautifulSoup(xml_content, 'xml')
        objs = bs.findAll('object')
        for obj in objs:
            obj_name = obj.findChildren('name')[0].text
            if len(obj.findChildren('difficult'))>0:
                difficult = int(obj.findChildren('difficult')[0].contents[0])
            else:
                difficult = -1
            if len(obj.findChildren('truncated'))>0:
                truncated = int(obj.findChildren('truncated')[0].contents[0])
            else:
                truncated = -1
            bbox = obj.findChildren('bndbox')[0]
            xmin = int(bbox.findChildren('xmin')[0].contents[0])
            ymin = int(bbox.findChildren('ymin')[0].contents[0])
            xmax = int(bbox.findChildren('xmax')[0].contents[0])
            ymax = int(bbox.findChildren('ymax')[0].contents[0])
            annotations.append(Entity(obj_name, xmin, xmax, ymin, ymax, difficult, truncated))
        return annotations

def get_image_list(dir, filename):
    image_list = open(os.path.join(dir, filename)).readlines()
    return [image_name.strip() for image_name in image_list]

def process_image(image_data, line_thickness=4):
    image = cv2.imread(image_data.image_path)
    image = cv2.putText(image, image_data.image_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for ann in image_data.annotations:
        box_color = (0, 255, 0)  #Green
        if ann.difficult or ann.truncated:
            box_color = (0, 0, 255) #Red
        image = cv2.rectangle(image, (ann.xmin, ann.ymin), (ann.xmax, ann.ymax), box_color, line_thickness)
        image = cv2.putText(image, ann.name, (ann.xmin, ann.ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def plot_dual_image(image, image2):
    ax = plt.subplot(121)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title("Image 1")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    ax = plt.subplot(122)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.title("Image 2")
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    plt.axis('off')
    plt.tight_layout()

    return plt.gcf()


def main():
    image_path1 = args.image_path1
    image_path2 = args.image_path2
    output_dir = args.output_dir
    annotation_dir1 = args.annotation_dir1
    annotation_dir2 = args.annotation_dir2

    image_dir1 = os.path.dirname(image_path1)
    image_filename1 = os.path.basename(image_path1)
    image_name1 = os.path.splitext(image_filename1)[0]
    image_dir2 = os.path.dirname(image_path2)
    image_filename2 = os.path.basename(image_path2)
    image_name2 = os.path.splitext(image_filename2)[0]

    image_data1 = Data(image_dir1, annotation_dir1, image_name1)
    image_data2 = Data(image_dir2, annotation_dir2, image_name2)

    image1 = process_image(image_data1, args.line_thickness)
    image2 = process_image(image_data2, args.line_thickness)

    fig = plot_dual_image(image1, image2)

    if args.output_dir:
        #cv2.imwrite(os.path.join(output_dir, image_name + ".jpg"), image)
        plt.savefig(os.path.join(output_dir, "dual_image_" + image_name1 + "_" + image_name2 + ".jpg"), dpi=600)
    #cv2.imshow('image', image)
    #plt.show()


if __name__ == "__main__":
    main()

    print("=== Program end ===")
