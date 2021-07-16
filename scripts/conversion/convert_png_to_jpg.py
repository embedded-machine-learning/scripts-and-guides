#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert PNG images to JPG.

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
# Source:

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
from PIL import Image
import glob

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = []
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Convert PNG images to JPG')
parser.add_argument("-id", '--image_dir',
                    default="images",
                    help='Image file directory', required=False)

args = parser.parse_args()
print(args)


def convert(image_dir):

    for image_path in glob.glob(image_dir + '/*.png'):
        print("Found PNG image: " + image_path)
        image_base_name = os.path.basename(image_path).split('.')[-2]
        im1 = Image.open(image_path)
        im1.save(image_dir + '/' + image_base_name + '.jpg')

if __name__ == "__main__":
    convert(args.image_dir)

    print("=== Program end ===")