#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# command to generate report with
# python3 /opt/intel/openvino_2020.4.287/deployment_tools/tools/benchmark_tool/benchmark_app.py \
# --path_to_model ~/projects/nuc_format/inference-demo/exported-models/tf2oda_efficientdetd0_512x384_pedestrian_LR02/saved_model/saved_model.xml \
# --report_type average_counters -niter 10 -d MYRIAD -nireq 1

# Data Formats
# Date Model Model_Short Framework Network Resolution Dataset Custom_Parameters Hardware Hardware_Optimization DetectionBoxes_Precision/mAP DetectionBoxes_Precision/mAP@.50IOU DetectionBoxes_Precision/mAP@.75IOU DetectionBoxes_Precision/mAP (small) DetectionBoxes_Precision/mAP (medium) DetectionBoxes_Precision/mAP (large) DetectionBoxes_Recall/AR@1 DetectionBoxes_Recall/AR@10 DetectionBoxes_Recall/AR@100 DetectionBoxes_Recall/AR@100 (small) DetectionBoxes_Recall/AR@100 (medium) DetectionBoxes_Recall/AR@100 (large)
# Date Model Model_Short Framework Network Resolution Dataset Custom_Parameters Hardware Hardware_Optimization Batch_Size Throughput Mean_Latency Latencies

# EXAMPLE USAGE - the following command extracts infos from reports and parses them into a new file
# python3 openvino_latency_parser.py g_--avrep tf_inceptionv1_224x224_imagenet_3.16G_avg_cnt_rep.csv --inf_rep tf_inceptionv1_224x224_imagenet_3.16G.csv --save_new

#EXAMPLE USAGE - the following command extracts infos from reports and appends them to a new line of the existing_file csv
# python3 openvino_latency_parser.py --avg_rep tf_inceptionv1_224x224_imagenet_3.16G_avg_cnt_rep.csv --inf_rep tf_inceptionv1_224x224_imagenet_3.16G.csv --existing_file latency_tf_inceptionv1_224x224_imagenet_3.16G.csv


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

"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import sys, os, json, argparse
import logging as log
from datetime import datetime

# Libs
import cv2
import numpy as np
import pandas as pd
from openvino.inference_engine import IECore

# Own modules

__author__ = 'Matvey Ivanov'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Matvey Ivanov']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Matvey Ivanov'
__email__ = 'matvey.ivanov@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description="NCS2 settings test")
parser.add_argument(
        "-m",
        "--model",
        default="./model.xml",
        help="model to test with",
        type=str,
        required=False,
    )
parser.add_argument(
        "-i",
        "--input",
        default="./input",
        help="images for the inference",
        type=str,
        required=False,
    )
args = parser.parse_args()
print(args)

if __name__ == "__main__":


    model_name = args.model.split("/")[-1:][
        0
    ]  # extract model name from parsed model path

    if not ".xml" in model_name:
        sys.exit("Invalid model xml given!")

    model_xml = args.model
    model_bin = args.model.split(".xml")[0] + ".bin"

    if not os.path.isfile(model_xml) or not os.path.isfile(model_bin):
        sys.exit("Could not find IR model for: " + model_xml)

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    print("Loaded model: {}, weights: {}".format(model_xml, model_bin))

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    in_blob = net.input_info[input_blob].input_data.shape
    net.input_info[input_blob].precision = "U8"
    net.batch_size = 1

    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    for i in range(n):
        image = cv2.imread(args.input[i])
        image_height, image_width = image.shape[:-1]
        images_hw.append((image_height), (image_width))
        if image.shape[:1] != (h, w):
            log.warning(
                "Image {} is resized from {} to {}".format(
                    args.input[i], image.shape[:-1], (h, w)
                )
            )
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        images[i] = image

    print("Loading network")
    exec_net = ie.load_network(network=net, device_name="MYRIAD", num_requests=1)

    print("Starting inference")
    res = exec_net.infer(inputs={input_blob: images})
    # print(res)
    print("\nType of result object", type(res))

    res = res[out_blob]
    data = res[0][0]
    combined_data = []
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            image_id = np.int(proposal[0])
            image_height, image_width = images_hw[image_id]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(image_width * proposal[3])
            ymin = np.int(image_height * proposal[4])
            xmax = np.int(image_width * proposal[5])
            ymax = np.int(image_height * proposal[6])
            if proposal[2] > 0.5:
                combination_str = (
                    str(proposal[0])
                    + " "
                    + str(image_width)
                    + " "
                    + str(image_height)
                    + " "
                    + str(label)
                    + " "
                    + str(xmin)
                    + " "
                    + str(ymin)
                    + " "
                    + str(xmax)
                    + " "
                    + str(ymax)
                    + " "
                    + str(confidence)
                )
                combined_data.append([combination_str.strip()])

    dataframe = pd.DataFrame(
        combined_data,
        columns=[
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "score",
        ],
    )
    dataframe.to_csv("output" + ".csv", index=False)
