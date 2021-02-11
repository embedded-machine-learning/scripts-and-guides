#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rename folder of images according a special pattern.

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
#from __future__ import print_function

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
parser.add_argument("-i", "--image_dir", default="samples/images", help="Directory with images", required=False)
parser.add_argument("-n", "--new_name", default="frame_", help="New name for images", required=False)
parser.add_argument("-t", "--no_types", action='store_true', default=False, help="If true, no types will be selected. "
                                                                                 "Else jpg and png.", required=False)
parser.add_argument("-s", "--start_number", type=int, default=0, help="Start number for renaming images", required=False)

args = parser.parse_args()

def renameImages(path, new_name, start_number, no_types):
    """ Renames all images in 'path' to a specific pattern.
        
    Inputs:
        - path: path of the firectory with the images.
        - new_name: string with the new name pattern for all files.
        Example: if new_name='cat', all the images will be renamed to
        'cat_1', 'cat_2', 'cat_3', etc.
    """
    # Types of file accepted by tensorflow API
    if no_types:
        types = ("*.*", "")
        print("All files in folder will be renamed.")
    else:
        types = ("*.jpg", "*.png")
        print("*.jpg and *.png will be renamed.")

    # Rename images
    idx = start_number
    for tp in types:
        if tp != "":
            for fpath in glob.glob(os.path.join(path, tp)):
                #os.rename(fpath, os.path.join(path, new_name + "_" + str(idx) + tp[1:]))
                file_ext = os.path.basename(fpath).split('.')[-1]
                new_filename = os.path.join(path, new_name + "{:04d}".format(idx) + "." + file_ext)
                os.rename(fpath, new_filename)
                print("Renamed {}->{}".format(fpath, new_filename))
                idx += 1
            
if __name__ == "__main__":
    # Path of the images

    renameImages(args.image_dir, args.new_name, args.start_number, args.no_types)

    print("Finished")