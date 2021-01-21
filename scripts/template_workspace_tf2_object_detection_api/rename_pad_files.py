#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rename files for extending padded numbers with 0000.

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


"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import os
import glob

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
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.5.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Rename image files')
parser.add_argument("-i", "--files_dir", default="samples/tmp", help="Directory with files to pad", required=False)
#parser.add_argument("-n", "--new_name", default="frame_", help="New name for images", required=False)
#parser.add_argument("-t", "--no_types", action='store_true', default=False, help="If true, no types will be selected. "
#                                                                                 "Else jpg and png.", required=False)

args = parser.parse_args()

def renameFiles(path):
    '''


    :param path: file dir path
    :return: Nothing
    '''

    for filename in os.listdir(path):
        # :-4 remove extension of file and split for _
        if len(filename[:-4].split('_'))<2:
            prefix=""
            num=filename[:-4].split('_')[-1]
        else:
            num = filename[:-4].split('_')[-1]
            prefix = filename[:-4].split('_')[-2]
        file_type = filename.split('.')
        num = num.zfill(4)

        if prefix != "":
            new_filename = prefix + "_" + num + "." + file_type[-1]
        else:
            new_filename = num + "." + file_type[-1]
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
        print("Renamed {}->{}".format(filename, new_filename))

if __name__ == "__main__":
    # Path of the images

    renameFiles(args.files_dir)