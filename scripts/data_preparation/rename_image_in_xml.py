#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rename image names within xml according to a defined schema.

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
# Source: https://gist.github.com/jinyu121/a222492405890ce912e95d8fb5367977

"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
import glob
import json
import os

# Libs
from tqdm import tqdm
from xmltodict import unparse
import argparse
import xml.etree.ElementTree as ET

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(description='For PASCAL VOC, rename image filename within xml files of a folder.')
parser.add_argument("-af", '--annotation_folder',
                    default="samples/tmp",
                    help='Annotation folder.', required=False)
parser.add_argument("-ip", '--image_prefix',
                    default="test_",
                    help='Imageprefix in front of the numbers, e.g. test_', required=False)
parser.add_argument("-o", '--output_folder',
                    default="samples/tmp/renamed",
                    help='Output folder for renamed files', required=False)
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


def rename_image_in_xml(annotation_folder, image_prefix, output_folder):
    '''


    '''

    #Output folder generation
    if output_folder != annotation_folder and not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        print("Created ", output_folder)
    else:
        print("output_folder is the same as the input folder. Replace files")

    #xml_file_list = []
    print("Filter not used. Select all xml files of the folder")
    xml_file_list = glob.glob(annotation_folder + '/*.xml')
    for xml_file in xml_file_list: #glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        #for member in root.findall('object'):

        old_file_name = root.find('filename').text

        file_ext = os.path.basename(old_file_name).split('.')[-1]
        if len(old_file_name[:-4].split('_'))<2:
            prefix=""
            num=old_file_name[:-4].split('_')[-1]
        else:
            num = old_file_name[:-4].split('_')[-1]
        num = num.zfill(4)
        new_filename = image_prefix + num + "." + file_ext

        root.find('filename').text = new_filename
        print("In {}, replace {} with {}.".format(xml_file, old_file_name, new_filename))

        #Save xml again
        target_path = os.path.join(output_folder, os.path.basename(xml_file))
        tree.write(target_path)
        print("File saved in {}".format(target_path))



if __name__ == "__main__":

    rename_image_in_xml(args.annotation_folder, args.image_prefix, args.output_folder)


    print("=== Program end ===")
