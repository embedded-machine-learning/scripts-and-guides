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
import warnings

# Libs
import glob
import pandas as pd
import argparse
from PIL import Image

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Alexander Wendt', 'https://gist.github.com/goodhamgupta']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Convert Yolo detections to Tensorflow detections csv file')
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

args = parser.parse_args()
print(args)


def read_annotation_file(annotation_filepath, image_dir):
    file_prefix = os.path.basename(annotation_filepath).split('.txt')[0]
    image_file_name = file_prefix + '.jpg'
    annotation_file_name = "{}.txt".format(file_prefix)
    # annotation_file_path = os.path.join(annotation_dir, annotation_file_name)
    image_file_path = os.path.join(image_dir, image_file_name)

    if not os.path.exists(image_file_path):
        df = None
        warnings.warn("{} does not exist. Skipping.".format(image_file_path))
    else:
        img = Image.open(image_file_path)
        w, h = img.size

        if os.path.exists(annotation_filepath):
            #print("Convert annotation {}".format(annotation_filepath))
            # with open(os.path.join(annotation_dir, )"labels/" + file_path, 'r') as file:

            # Create DF for tf csv
            df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])

            with open(os.path.join(annotation_filepath), 'r') as file:
                print("Process ", annotation_filepath)
                lines = file.readlines()
                voc_labels = []
                for line in lines:
                    voc = []
                    line = line.strip()
                    data = line.split()
                    # voc.append(CLASS_MAPPING.get(data[0]))

                    bbox_width = float(data[3]) * w
                    bbox_height = float(data[4]) * h
                    center_x = float(data[1]) * w
                    center_y = float(data[2]) * h
                    xmin = (round(center_x - (bbox_width / 2))) / w
                    ymin = (round(center_y - (bbox_height / 2))) / h
                    xmax = (round(center_x + (bbox_width / 2))) / w
                    ymax = (round(center_y + (bbox_height / 2))) / h

                    if data[5] is not None:
                        score = data[5]
                    else:
                        raise Exception("No score is given: {}".format(image_file_name))

                    new_row = {'filename': image_file_name,
                               'width': w,
                               'height': h,
                               'class': str(int(data[0]) + 1),
                               # TF Detections classes always start by 1 and not by 0 as Yolo
                               'xmin': xmin,
                               'ymin': ymin,
                               'xmax': xmax,
                               'ymax': ymax,
                               'score': score}

                    df = df.append(new_row, ignore_index=True)
                    #print("Added ", new_row)
                    # voc.append(classes_dict.get(data[0]))
                    # voc_labels.append(voc)
                # create_file(file_prefix, w, h, voc_labels, target_annotation_dir)
        # elif create_empty_images:
        #    print("Annotation does not exist {}. Create empty annotation".format(annotation_file_path))
        #    voc_labels = []
        #    create_file(file_prefix, w, h, voc_labels, target_annotation_dir)
        else:
            print("Annotation does not exist {}. Do nothing".format(annotation_filepath))

    return df


def convert_yolo_to_tfcsv(annotation_dir, image_dir, output_path):
    """


    """
    # Make output dir if not existent
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    detections_df = pd.DataFrame(
        columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])

    for annotation_filepath in glob.glob(annotation_dir + '/*.txt'):
        # Check if images exist

        # Load detection files
        single_annotation_df = read_annotation_file(annotation_filepath, image_dir)

        if single_annotation_df is not None:
            ##print("File parsed")
            detections_df = detections_df.append(single_annotation_df, ignore_index=True)
        else:
            print("{} has no corresponding image. Continue.".format(annotation_filepath))
            continue

    detections_df.set_index(['filename'], inplace=True)
    detections_df.to_csv(output_path, sep=';', header=True)


if __name__ == "__main__":
    convert_yolo_to_tfcsv(args.annotation_dir, args.image_dir, args.output)

    print("=== Program end ===")
