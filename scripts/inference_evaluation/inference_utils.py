#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Library with methods for inference

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

# The following script uses several method fragments from Tensorflow
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

Tensorflow has the following licence:
# ==============================================================================
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports


# Libs


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


def get_info_from_modelname(model_name, model_short_name=None, model_optimizer_prefix=['TRT', 'OV']):
    '''
    Extract information from file name

    :argument

    :return

    '''
    info = dict()

    info['model_name'] = model_name
    info['framework'] = str(model_name).split('_')[0]
    info['network'] = str(model_name).split('_')[1]
    info['resolution'] = list(map(int, (str(model_name).split('_')[2]).split('x')))
    info['dataset'] = str(model_name).split('_')[3]
    info['hardware_optimization'] = ""
    info['custom_parameters'] = ""
    custom_list = []
    if len(model_name.split("_", 4)) > 4:
        rest_parameters = model_name.split("_", 4)[4]

        for r in rest_parameters.split("_"):
            #FIXME: Make a general if then for this, not just the 2 first entries in the list
            if str(r).startswith(model_optimizer_prefix[0]) or str(r).startswith(model_optimizer_prefix[1]):
                info['hardware_optimization'] = r
            else:
                custom_list.append(r)
                # if info['custom_parameters'] == "":
                #    info['custom_parameters'] = r
                # else:
                #    info['custom_parameters'] = info['custom_parameters'] + "_" + r

    info['custom_parameters'] = str(custom_list)

    # Enhance inputs
    if model_short_name is None:
        info['model_short_name'] = model_name
        print("No short models name defined. Using the long name: ", model_name)
    else:
        info['model_short_name'] = model_short_name

    return info
