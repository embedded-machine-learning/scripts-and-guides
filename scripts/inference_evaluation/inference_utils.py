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
from datetime import datetime
import logging
import os

# Libs
import pandas as pd
import numpy as np

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

log = logging.getLogger()
stdout=logging.StreamHandler()
#formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
formatter = logging.Formatter('%(message)s')
stdout.setFormatter(formatter)
log.addHandler(stdout)
log.setLevel(logging.DEBUG)


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

def save_latencies_to_csv(latencies, batch_size, number_runs, hardware_name, model_name, model_short_name, latency_out, index=None):
    '''
    Save a list of latencies to csv file

    :argument
        latencies: List of latencies of single measurements
        batch_size: Used batch size
        number_runs: Number of runs of the network, should be len(latencies) for single measurements or an integer for
            mean values
        hardware_name: Hardware name e.g. NUC
        model_name: Model name extracted from a file
        model_short_name: Short model name
        latency_out: Filename of the output latency file. If the file already exists, the result is appended
        index=None: Generated index that is used to link multiple measurements

    :return
        None

    '''

    # Get model info
    model_info = get_info_from_modelname(model_name, model_short_name)

    # If
    if not index:
        index = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_name
        print("No index was provided. Create index ", index)

    # Calucluate mean latency
    mean_latency = np.array(latencies).mean()

    # Calulate throughput
    # throughput = 1 / mean_latency
    throughput = number_runs * batch_size / np.array(latencies).sum()

    if len(latencies)>1:
        latency_string = str(latencies)
        log.debug("Single strings available")
    else:
        latency_string = None
        log.debug("A mean value was provided.")

    # Save latencies
    print("Mean inference time: {}".format(mean_latency))
    series_index = ['Index',
                    'Date',
                    'Model',
                    'Model_Short',
                    'Framework',
                    'Network',
                    'Resolution',
                    'Dataset',
                    'Custom_Parameters',
                    'Hardware',
                    'Hardware_Optimization',
                    'Batch_Size',
                    'Throughput',
                    'Mean_Latency',
                    'Latencies']
    content = [index,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               model_info['model_name'],
               model_info['model_short_name'],
               model_info['framework'],
               model_info['network'],
               str(model_info['resolution']),
               model_info['dataset'],
               str(model_info['custom_parameters']),
               hardware_name,
               str(model_info['hardware_optimization']),
               1,
               throughput,
               mean_latency,
               latency_string]
    # Create DataFrame
    df = pd.DataFrame([pd.Series(data=content, index=series_index, name="data")])
    df.set_index('Index', inplace=True)
    # Append dataframe wo csv if it already exists, else create new file
    if os.path.isfile(latency_out):
        old_df = pd.read_csv(latency_out, sep=';')
        old_df['Custom_Parameters'] = old_df['Custom_Parameters'].replace(np.nan, '', regex=True)
        old_df['Model_Short'] = old_df['Model_Short'].replace(np.nan, '', regex=True)
        old_df['Hardware_Optimization'] = old_df['Hardware_Optimization'].replace(np.nan, '', regex=True)
        old_df['Latencies'] = old_df['Latencies'].replace(np.nan, '', regex=True)

        print("Old DF types: ", old_df.dtypes)
        print("New DF types: ", df.dtypes)

        merged_df = old_df.reset_index().merge(df.reset_index(), how="outer").set_index('Index').drop(
            columns=['index'])  # pd.merge(old_df, df, how='outer')

        merged_df.to_csv(latency_out, mode='w', header=True, sep=';')
        # df.to_csv(latency_out, mode='a', header=False, sep=';')
        print("Appended evaluation to ", latency_out)
    else:
        df.to_csv(latency_out, mode='w', header=True, sep=';')
        print("Created new measurement file ", latency_out)


def convert_reduced_detections_tf2_to_df(image_filename, image_np, boxes, classes, scores, min_score=0.5):
    '''
    Convert TF2 detections for one image to df, which can then be merged with the scores of many images
    and saved as detections.csv

    :param image_filename: File name
    :param image_np: Image as array
    :param boxes: Boxes from TF2 detections
    :param classes: Classes from TF2 detections
    :param scores: Scores from TF2 detections
    :param min_score: Min score to pass to the dataframe

    :return: dataframe with detections
    '''
    image_width = image_np.shape[1]
    image_height = image_np.shape[0]

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
    xml_df = pd.DataFrame(columns=column_name)

    for i in range(scores.shape[0]):
        if min_score <= scores[i]:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            content = [image_filename, image_width, image_height,
                       classes[i], xmin, ymin, xmax, ymax, scores[i]]
            xml_df = xml_df.append(pd.DataFrame([content], columns=column_name))

    return xml_df

def generate_measurement_index(model_name):
    '''
    Generate an index for a measurement that is used as a database key.

    :param model_name: Model name long
    :return: index
    '''
    index = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_name
    return index