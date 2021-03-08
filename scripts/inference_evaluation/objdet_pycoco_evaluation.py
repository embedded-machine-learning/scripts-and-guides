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

# Libs
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd

# If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
#matplotlib.use('TkAgg')

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
args = parser.parse_args()


def evaluate_inference(coco_gt_file, coco_det_file, output_file, model_name, hardware_name, model_short_name=None):
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

    # Create df for file export
    content = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, model_short_name, hardware_name]

    series_index = ['Date',
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

    framework = str(model_name).split('_')[0]
    network = str(model_name).split('_')[1]
    resolution = str(model_name).split('_')[2]
    dataset = str(model_name).split('_')[3]
    if (len(model_name.split("_", 4))>4):
        custom_parameters = model_name.split("_", 4)[4]
    else:
        custom_parameters = ""

    content = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               model_name,
               model_short_name,
               framework,
               network,
               resolution,
               dataset,
               custom_parameters,
               hardware_name,
               'None'
               ]

    content.extend(cocoEval.stats)

    # Create DataFrame
    df = pd.DataFrame([pd.Series(data=content, index=series_index, name="data")])
    df.set_index('Date', inplace=True)

    # Append dataframe wo csv if it already exists, else create new file
    if os.path.isfile(output_file):
        old_df = pd.read_csv(output_file, sep=';')

        merged_df = old_df.reset_index().merge(df.reset_index(), how="outer").set_index('Date').drop(
            columns=['index'])  # pd.merge(old_df, df, how='outer')

        merged_df.to_csv(output_file, mode='w', header=True, sep=';')
        # df.to_csv(latency_out, mode='a', header=False, sep=';')
        print("Appended evaluation to ", output_file)
    else:
        df.to_csv(output_file, mode='w', header=True, sep=';')
        print("Created new measurement file ", output_file)



if __name__ == "__main__":

    evaluate_inference(args.groundtruth_file, args.detection_file, args.output_file,
                       args.model_name, args.hardware_name, model_short_name=args.model_short_name)

    print("=== Program end ===")