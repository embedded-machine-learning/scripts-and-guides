#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert keras h5 models to TF2 saved_model

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
# Source:

"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import json
import os
import argparse
import time
import warnings
from datetime import datetime
import logging

# Libs
import tensorflow as tf

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

parser = argparse.ArgumentParser(description='Convert Keras to Tensorflow 2 Saved Model')
parser.add_argument("-i", '--input_path', default=None,
                    help='h5 model path', required=False)
parser.add_argument("-o", '--output_dir', default=None,
                    help='Saved model path', required=False)
args = parser.parse_args()

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

log.info(args)



#from custom_layer import CustomLayer
#model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer})


def convert(input_path, output_dir):
    model = tf.keras.models.load_model(input_path, compile=False)  #Compile=False means no training model
    tf.saved_model.save(model,output_dir)

if __name__ == "__main__":
    convert(args.input_path, args.output_dir)

    print("=== Program end ===")