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
import time
from datetime import datetime

# Libs
import cv2
import numpy as np
import pandas as pd
from openvino.inference_engine import IECore

# Own modules

__author__ = "Matvey Ivanov"
__copyright__ = (
    "Copyright 2021, Christian Doppler Laboratory for " "Embedded Machine Learning"
)
__credits__ = ["Matvey Ivanov"]
__license__ = "ISC"
__version__ = "0.1.0"
__maintainer__ = "Matvey Ivanov"
__email__ = "matvey.ivanov@tuwien.ac.at"
__status__ = "Experiental"

parser = argparse.ArgumentParser(description="NCS2 settings test")
parser.add_argument(
    "-m",
    "--model_path",
    default="./model.xml",
    help="model to test with",
    type=str,
    required=False,
)
parser.add_argument(
    "-i",
    "--image_dir",
    default="./images",
    help="images for the inference",
    type=str,
    required=False,
)
parser.add_argument(
    "-d",
    "--device",
    default="CPU",
    help="target device to run the inference [CPU, GPU, MYRIAD]",
    type=str,
    required=False,
)

parser.add_argument(
    "-out",
    '--detections_out',
    default='detections.csv',
    help='Output file detections',
    type=str,
    required=False
)

parser.add_argument(
    "-ls",
    '--labels_start',
    default=0,
    help='Start value of the classes. For TF2ODA, the start value is 0. For AutoML EfficientDet, the start value is 1.'
         'Look into the detections file and compare with ground truth if the metric passes bad 0 match although the '
         'bounding boxes were correctly drawed. In such a case, the labels_start might be 1 and not 0.',
    type=int,
    required=False
)

args = parser.parse_args()
print(args)

if __name__ == "__main__":

    model_name = args.model_path.split("/")[-1:][
        0
    ]  # extract model name from parsed model path

    if not ".xml" in model_name:
        sys.exit("Invalid model xml given!")

    model_xml = args.model_path
    model_bin = args.model_path.split(".xml")[0] + ".bin"

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

    print("Loading network and perform on ", args.device)
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=1)

    combined_data = []
    _, _, net_h, net_w = net.input_info[input_blob].input_data.shape

    for filename in os.listdir(args.image_dir):

        total_latency_start_time = time.time()
        original_image = cv2.imread(os.path.join(args.image_dir, filename))
        image = original_image.copy()

        if image.shape[:-1] != (net_h, net_w):
            log.debug(f"Image {args.image_dir} is resized from {image.shape[:-1]} to {(net_h, net_w)}")
            image = cv2.resize(image, (net_w, net_h))

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        print("Starting inference for picture: " + filename)
        res = exec_net.infer(inputs={input_blob: image})

        # print(res)
        # print("Type of result object", type(res))

        output_image = original_image.copy()
        h, w, _ = output_image.shape

        if len(net.outputs) == 1:
            res = res[out_blob]
            # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
            # where N is the number of detected bounding boxes
            detections = res.reshape(-1, 7)
        else:
            detections = res["boxes"]
            labels = res["labels"]
            # Redefine scale coefficients
            w, h = w / net_w, h / net_h

        for i, detection in enumerate(detections):
            combination_str = ""
            if len(net.outputs) == 1:
                _, class_id, confidence, xmin, ymin, xmax, ymax = detection
            else:
                class_id = labels[i]
                xmin, ymin, xmax, ymax, confidence = detection

            if confidence > 0.5:
                label = int(class_id) + args.labels_start
                xmin = float(xmin)
                ymin = float(ymin)
                xmax = float(xmax)
                ymax = float(ymax)
                combined_data.append(
                    [
                        filename,
                        str(w),
                        str(h),
                        str(label),
                        str(xmin),
                        str(ymin),
                        str(xmax),
                        str(ymax),
                        str(confidence),
                    ]
                )

        total_latency_stop_time = time.time()
        total_latency = total_latency_stop_time - total_latency_start_time
        print("Total latency for {} : {}s".format(filename, total_latency))

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

    # Warn if no detections
    print(detections)

    # Create output directories
    if not os.path.isdir(os.path.dirname(args.detections_out)):
        os.makedirs(os.path.dirname(args.detections_out))
        print("Created ", os.path.dirname(args.detections_out))

    dataframe.to_csv(args.detections_out, index=False, sep=";")  #
    print("Written detections to ", args.detections_out)
