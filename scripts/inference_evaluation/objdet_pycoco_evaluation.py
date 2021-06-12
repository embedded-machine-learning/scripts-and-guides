#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate validation data and coco data for object detection with the COCO evaluation metrics.

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

parser = argparse.ArgumentParser(description='Pycoco Tools Evaluator')
parser.add_argument("-gt", '--groundtruth_file', default='gt.json',
                    help='Coco ground truth path', required=False)
parser.add_argument("-det", '--detection_file', default='det.json',
                    help='Coco detections path', required=False)
parser.add_argument("-o", '--output_file', default=None,
                    help='Save/appends results to an output file', required=False)
parser.add_argument("-m", '--model_name', default='Default_Network',
                    help='Add model name', required=False)
parser.add_argument("-ms", '--model_short_name', default=None, type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-hw", '--hardware_name', default='Default_Hardware',
                    help='Add hardware name', required=False)
parser.add_argument('-id', '--index_save_file', type=str, default='./tmp/index.txt',
                    help='Use the string in the index file as a key for the measurement to combine it with latency '
                         'measurements.', required=False)
args = parser.parse_args()

log = logging.getLogger()
stdout=logging.StreamHandler()
#formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
formatter = logging.Formatter('%(message)s')
stdout.setFormatter(formatter)
log.addHandler(stdout)
log.setLevel(logging.DEBUG)

log.info(args)

def evaluate_inference(coco_gt_file, coco_det_file, output_file, model_name, hardware_name, model_short_name=None,
                       index_save_file="./tmp/index.txt"):
    '''
    Format the results: https://cocodataset.org/#format-results
    Helping source: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/coco_evaluation.html

    '''

    if not os.path.isdir(os.path.dirname(coco_det_file)):
        os.makedirs(os.path.dirname(coco_det_file))
        print("Created ", os.path.dirname(coco_det_file))

    #Enhance inputs
    if model_short_name is None:
        model_short_name=model_name
        print("No short models name defined. Using the long name: ", model_name)


    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here 1: Bounding box
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    print('Running evaluation for *%s* results.' % (annType))

    # initialize COCO ground truth api
    #dataDir = '../'
    #dataType = 'val2014'
    #annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    print("Load ground truth file: ", coco_gt_file)
    cocoGt = COCO(coco_gt_file)

    # initialize COCO detections api
    #resFile = '%s/results/%s_%s_fake%s100_results.json'
    #resFile = resFile % (dataDir, prefix, dataType, annType)
    print("Load detection file: ", coco_det_file)
    cocoDt = cocoGt.loadRes(coco_det_file)

    imgIds = sorted(cocoGt.getImgIds())
    #imgIds = imgIds[0:100]
    #imgId = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    #Save the Coco evaluation to a csv file
    save_evalution_to_csv(cocoEval, hardware_name, model_name, model_short_name, output_file, index_save_file)


def save_evalution_to_csv(cocoEval, hardware_name, model_name, model_short_name, output_file, index_save_file=None):
    '''
    Save the Coco evaluation to a csv file

    :param cocoEval:
    :param hardware_name:
    :param index_save_file:
    :param model_name:
    :param model_short_name:
    :param output_file:
    :return:
    '''

    # Load index file
    if not os.path.exists(index_save_file):
        log.warning("Index file does not exist: {}. Generate a new index.".format(index_save_file))
        index = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_name
        print("No index was provided. Create index ", index)
    else:
        with open(index_save_file) as f:
            index = f.readline()
        log.debug("Load index from file: {}. Value: {}".format(index_save_file, index))
        
        #Remove the index file as a new inex shall be used next time
        os.remove(index_save_file)
        log.debug("Temporary index file removed from {}".format(index_save_file))


    # Create df for file export
    content = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, model_short_name, hardware_name]
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
                    'DetectionBoxes_Precision/mAP',
                    'DetectionBoxes_Precision/mAP@.50IOU',
                    'DetectionBoxes_Precision/mAP@.75IOU',
                    'DetectionBoxes_Precision/mAP (small)',
                    'DetectionBoxes_Precision/mAP (medium)',
                    'DetectionBoxes_Precision/mAP (large)',
                    'DetectionBoxes_Recall/AR@1',
                    'DetectionBoxes_Recall/AR@10',
                    'DetectionBoxes_Recall/AR@100',
                    'DetectionBoxes_Recall/AR@100 (small)',
                    'DetectionBoxes_Recall/AR@100 (medium)',
                    'DetectionBoxes_Recall/AR@100 (large)']
    model_info = inf.get_info_from_modelname(model_name, model_short_name)

    content = [index,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               model_name,
               model_short_name,
               model_info['framework'],
               model_info['network'],
               str(model_info['resolution']),
               model_info['dataset'],
               str(model_info['custom_parameters']),
               hardware_name,
               str(model_info['hardware_optimization'])
               ]
    content.extend(cocoEval.stats)
    # Create DataFrame
    df = pd.DataFrame([pd.Series(data=content, index=series_index, name="data")])
    df.set_index('Index', inplace=True)
    # Append dataframe wo csv if it already exists, else create new file
    if os.path.isfile(output_file):
        old_df = pd.read_csv(output_file, sep=';')
        old_df['Custom_Parameters'] = old_df['Custom_Parameters'].replace(np.nan, '', regex=True)
        old_df['Model_Short'] = old_df['Model_Short'].replace(np.nan, '', regex=True)
        # old_df['Custom_Parameters'] = old_df['Custom_Parameters'].astype(str)
        old_df['Hardware_Optimization'] = old_df['Hardware_Optimization'].replace(np.nan, '', regex=True)

        merged_df = old_df.reset_index().merge(df.reset_index(), how="outer").set_index('Index').drop(
            columns=['index'])  # pd.merge(old_df, df, how='outer')

        merged_df.to_csv(output_file, mode='w', header=True, sep=';')
        # df.to_csv(latency_out, mode='a', header=False, sep=';')
        print("Appended evaluation to ", output_file)
    else:
        df.to_csv(output_file, mode='w', header=True, sep=';')
        print("Created new measurement file ", output_file)


if __name__ == "__main__":

    evaluate_inference(args.groundtruth_file, args.detection_file, args.output_file,
                       args.model_name, args.hardware_name, model_short_name=args.model_short_name,
                       index_save_file=args.index_save_file)

    print("=== Program end ===")