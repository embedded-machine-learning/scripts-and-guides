#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge all kinds of validation data from several records into a single results file, e.g. latency, detection boxes
or power measurements. Merge by the file Index in each csv file.

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
# Source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import argparse
import os
import warnings
from datetime import datetime
import logging

# Libs
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np

# Own modules
import inference_utils as inf

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(description='Merge all kinds of validation data from several records into a single '
                                             'results file, e.g. latency, detection boxes or power measurements')
parser.add_argument("-lat", '--latency_file', default=None,
                    help='Latency path', required=False)
parser.add_argument("-coc", '--coco_eval_file', default=None,
                    help='Coco evaluation path', required=False)
parser.add_argument("-o", '--output_file', default="./results.csv",
                    help='Save/appends results to an output file', required=False)
args = parser.parse_args()

log = logging.getLogger()
stdout=logging.StreamHandler()
#formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
formatter = logging.Formatter('%(lineno)d:\t - %(message)s')
stdout.setFormatter(formatter)
log.addHandler(stdout)
log.setLevel(logging.DEBUG)

log.info(args)

def merge_results(latency_file, coco_eval_file, output_file):
    '''
    Merge results from latency and coco evaluations by the key Index

    :param latency_file:
    :param coco_eval_file:
    :param output_file:
    :return:
    '''

    #Create result file path if not existent
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    #load latency file
    if not os.path.exists(latency_file):
        raise Exception(latency_file + " does not exist. Latency measurements must be available. Quit program")
    latency = pd.read_csv(latency_file, sep=';')

    #Load coco eval file
    if not os.path.exists(coco_eval_file):
        raise Exception(latency_file + " does not exist. Performance measurements must be available. Quit program")
    performance = pd.read_csv(coco_eval_file, sep=';')

    #Merge
    latency_reduced = latency.drop(columns=['Date']).set_index(['Index'])
    latency_reduced = latency_reduced[~latency_reduced.index.duplicated(keep='first')]
    performance_reduced = performance.drop(columns=['Date']).set_index(['Index'])
    performance_reduced = performance_reduced[~performance_reduced.index.duplicated(keep='first')]

    lat_perf_df = pd.merge(latency_reduced, performance_reduced,
                           how='inner', left_index=True, right_index=True,
                           suffixes=("", "_extra"))
    #lat_perf_df = lat_perf_df.reset_index()
    lat_perf_df = lat_perf_df.loc[:, ~lat_perf_df.columns.str.endswith("_extra")]
    print("Available columns: ", lat_perf_df.columns)

    # Append dataframe wo csv if it already exists, else create new file
    if os.path.isfile(output_file):
        old_df = pd.read_csv(output_file, sep=';')
        old_df['Custom_Parameters'] = old_df['Custom_Parameters'].replace(np.nan, '', regex=True)
        old_df['Model_Short'] = old_df['Model_Short'].replace(np.nan, '', regex=True)
        old_df['Hardware_Optimization'] = old_df['Hardware_Optimization'].replace(np.nan, '', regex=True)
        old_df['Latencies'] = old_df['Latencies'].replace(np.nan, '', regex=True)

        lat_perf_df['Hardware_Optimization'] = lat_perf_df['Hardware_Optimization'].replace(np.nan, '', regex=True)
        lat_perf_df['Latencies'] = lat_perf_df['Latencies'].replace(np.nan, '', regex=True)
        lat_perf_df['Model_Short'] = lat_perf_df['Model_Short'].replace(np.nan, '', regex=True)
        lat_perf_df['Custom_Parameters'] = lat_perf_df['Custom_Parameters'].replace(np.nan, '', regex=True)

        print("Old DF types: {}. \nLatencies: {}".format(old_df.dtypes, old_df['Latencies']))
        print("New DF types: {}. \nLatencies: {}".format(lat_perf_df.dtypes, lat_perf_df['Latencies']))

        merged_df = old_df.reset_index().merge(lat_perf_df.reset_index(), how="outer").set_index('Index').drop(
            columns=['index'])  # pd.merge(old_df, df, how='outer')

        #Drop duplicated indices
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

        merged_df.to_csv(output_file, mode='w', header=True, sep=';')
        # df.to_csv(latency_out, mode='a', header=False, sep=';')
        print("Appended evaluation to ", output_file)
    else:
        lat_perf_df.to_csv(output_file, mode='w', header=True, sep=';')
        print("Created new measurement file ", output_file)


if __name__ == "__main__":

    merge_results(args.latency_file, args.coco_eval_file, args.output_file)

    print("=== Program end ===")