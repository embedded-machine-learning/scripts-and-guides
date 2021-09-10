# this script takes a neural network in the intermediate representation .pd
# and converts it to a Movidius NCS2 conform format with .xml and .bin
# runs inference on the generated model

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script takes a neural network Movidius NCS2 conform format with .xml and .bin
runs inference on the generated model with OpenVino

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
import os
from absl import flags, app
import sys
import time
from datetime import datetime

# Libs
import argparse
import numpy as np
import glob
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import matplotlib
from six import BytesIO
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

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

FLAGS = flags.FLAGS
flags.DEFINE_string('pb', 'yolov3.pb', 'intermediade representation')
flags.DEFINE_string('xml', 'yolov3.xml', 'movidius representation')
flags.DEFINE_string('save_folder', './tmp/', 'folder to save the resulting files')
flags.DEFINE_string('api', 'sync', 'synchronous or asynchronous mode [sync, async]')
flags.DEFINE_string('niter', '100', 'number of iterations, useful in async mode')
flags.DEFINE_string('hw', 'MYRIAD', 'MYRIAD/CPU')
flags.DEFINE_string('batch_size', '1', 'Batch size')
flags.DEFINE_string('nireq', '1', 'Number of parallel requests in async mode')
flags.DEFINE_string('size', '[1,224,224,3]', '[1,224,224,3]')
flags.DEFINE_string('openvino_path', '/opt/intel/openvino', 'OpenVino path')
flags.DEFINE_string('output_dir', 'profiling_data', 'Report output directory')
print(flags.FLAGS)




def perform_inference(bench_app_file, is_linux, xml_path):
    '''
    Perform inference with an OpenVino model




    '''
    model_name = xml_path.split(".xml")[0].split("/")[-1]
    # benchmark_app inference
    niter = FLAGS.niter
    api = FLAGS.api
    report_dir = FLAGS.output_dir  # "profiling_data"
    if not os.path.isdir(os.path.dirname(report_dir)):
        os.makedirs(os.path.dirname(report_dir))
        print("Created ", os.path.dirname(report_dir))

    #if api == "sync":
    #    report_dir += "_sync"
    #    niter_str = ""
    #elif api == "async":
    #    report_dir += "_async_" #+ str(niter)
    #    niter_str = str(niter)
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    if is_linux:
        c_bench = ("python3 " + bench_app_file +
                   " --path_to_model " + xml_path)

    else:
        c_bench = ("python " + "\"" + bench_app_file + "\"" +
                   " --path_to_model " + "\"" + xml_path + "\"")
                   # " -d " + FLAGS.hw +
                   # " -b " + FLAGS.batch_size +
                   # " -api " + FLAGS.api +
                   # # " --exec_graph_path " + os.path.join(graph_dir, "graph") +
                   # " -niter " + str(niter) +
                   # " --report_type average_counters" +
                   # " --report_folder " + report_dir)

    c_bench = c_bench + (" -d " + FLAGS.hw +
                   " -b " + FLAGS.batch_size +
                   " -api " + FLAGS.api +
                   # " --exec_graph_path " + os.path.join(graph_dir, "graph") +
                   " -niter " + str(niter) +
                   " -pc " +
                   " -nireq " + FLAGS.nireq +
                   " --report_type average_counters" +
                   " --report_folder " + report_dir)
    # c_bench = ("python3 " + bench_app_file +
    # " -m "  + xml_path +
    # " -d " + FLAGS.hw +
    # #" -b 1 " +
    # " -api " + FLAGS.api +
    # #" --exec_graph_path " + os.path.join(graph_dir, "graph") +
    # " -niter " + str(niter) +
    # " --report_type average_counters" +
    # " --report_folder " + report_dir)
    if os.system(c_bench):
        sys.exit("An error has occured during benchmarking!")
    # rename the default report file name: Generate default names
    report_long_name = os.path.join(report_dir, "benchmark_average_counters_report.csv")
    report_long_target_name = os.path.join(report_dir, "benchmark_average_counters_report_" +FLAGS.hw + "_" + api + ".csv")
    report_short_name = os.path.join(report_dir, "benchmark_report.csv")
    report_short_target_name = os.path.join(report_dir, "benchmark_report_" + FLAGS.hw + "_" + api + ".csv")
    # Delete file if it already exists
    if os.path.isfile(report_long_target_name):
        os.remove(report_long_target_name)
    if os.path.isfile(report_short_target_name):
        os.remove(report_short_target_name)
    # Rename the general file into the new file name
    if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
        os.rename(report_long_name, report_long_target_name)
    if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
        os.rename(report_short_name, report_short_target_name)

    print("Saved reports to ", report_dir)
    print("**********REPORTS GATHERED**********")
    # # rename the default report file name
    # if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
    #     os.rename(os.path.join(report_dir, "benchmark_average_counters_report.csv"),
    #     os.path.join(report_dir, "benchmark_average_counters_report_" +
    #     model_name.split(".pb")[0] + "_" + FLAGS.hw + "_" + str(api) + niter_str + ".csv"))
    # if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
    #     os.rename(os.path.join(report_dir, "benchmark_report.csv"),
    #     os.path.join(report_dir, "benchmark_report_" +
    #     model_name.split(".pb")[0] + "_" + FLAGS.hw + "_" + str(api) + niter_str +  ".csv"))


def convert_model(mo_file):
    '''
    Convert models to OpenVino intermediate format

    :param mo_file:

    :return:
    '''

    # yolov3/yolov3-tiny json file necessary for conversion
    conv_cmd_str = ""
    if "yolov3-tiny" in FLAGS.pb or "yolov3-tiny" in FLAGS.xml:
        conv_cmd_str = (" --tensorflow_use_custom_operations_config" +
                        " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ yolo_v3_tiny.json")
    elif "yolov3" in FLAGS.pb or "yolov3-tiny" in FLAGS.xml:
        conv_cmd_str = (" --tensorflow_use_custom_operations_config" +
                        " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ yolo_v3.json")
    if "tf_" in FLAGS.pb:
        # Tensorflow conversion
        # input_shape for tensorflow : batch, width, height, channels
        shape = "[1," + FLAGS.pb.split("tf_")[1].split("_")[2] + "," + FLAGS.pb.split("tf_")[1].split("_")[3] + ",3]"

        c_conv = ("python3 " + mo_file +
                  " --input_model " + FLAGS.pb +
                  " --output_dir " + FLAGS.save_folder +
                  " --data_type FP16 " +
                  " --input_shape " + shape +
                  conv_cmd_str)
        xml_path = os.path.join(FLAGS.save_folder, FLAGS.pb.split(".pb")[0].split("/")[-1] + ".xml")
    elif "cf_" in FLAGS.pb or "dk_" in FLAGS.pb:
        # Caffe or Darknet conversion
        # input_shape : batch, channels, width, height
        input_proto = FLAGS.pb.split("/deploy.caffemodel")[0] + "/deploy.prototxt"
        if "cf_" in FLAGS.pb:
            shape = "[1,3," + FLAGS.pb.split("cf_")[1].split("_")[2] + "," + FLAGS.pb.split("cf_")[1].split("_")[
                3] + "]"
        elif "dk" in FLAGS.pb:
            shape = "[1,3," + FLAGS.pb.split("dk_")[1].split("_")[2] + "," + FLAGS.pb.split("dk_")[1].split("_")[
                3] + "]"

        if "SPnet" in FLAGS.pb:
            input_node = "demo"
        else:
            input_node = "data"

        c_conv = ("python3 " + mo_file +
                  " --input_model " + FLAGS.pb +
                  " --input_proto " + input_proto +
                  " --output_dir " + FLAGS.save_folder +
                  " --data_type FP16 " +
                  " --input_shape " + shape +
                  " --input " + input_node +  # input node sometimes called demo
                  conv_cmd_str)
    else:
        # Tensorflow conversion
        # input_shape for tensorflow : batch, width, height, channels
        # shape = "[1,513,1025,3]"

        c_conv = ("python3 " + mo_file +
                  " --input_model " + FLAGS.pb +
                  " --output_dir " + FLAGS.save_folder +
                  " --data_type FP16 " +
                  " --input_shape " + FLAGS.size +
                  " --input x" +
                  " --output Identity" +
                  conv_cmd_str)
        xml_path = os.path.join(FLAGS.save_folder, FLAGS.pb.split(".pb")[0].split("/")[-1] + ".xml")
    if os.system(c_conv):
        sys.exit("\nAn error has occured during conversion!\n")
    # set framework string and model name deploy.pb/forzen.pb
    framework = ""
    if "tf_" in FLAGS.pb:
        framework = "tf_"
        default_name = "frozen."
    elif "cf_" in FLAGS.pb:
        framework = "cf_"
        default_name = "deploy."
    elif "dk_" in FLAGS.pb:
        framework = "dk_"
        default_name = "deploy."
    else:
        framework = "tf_"
        default_name = "frozen_model."
    model_name = FLAGS.pb.split("/")[-1].split(".pb")[0]
    # rename all three generated files
    extension_list = ["xml", "bin", "mapping"]
    for ex in extension_list:
        os.rename(os.path.join(FLAGS.save_folder, model_name + "." + ex),
                  os.path.join(FLAGS.save_folder, framework + model_name + "." + ex))
    xml_path = os.path.join(FLAGS.save_folder, framework + model_name + ".xml")
    return xml_path

def main(argv):

    # Check if Windows oder Linux
    is_linux = True
    if (os.name == 'nt'):
        is_linux = False
        print("Windows system. Use windows specifics")
    else:
        print("Linux system. Use Linux specifics")

    bench_app_file = os.path.join(FLAGS.openvino_path, "deployment_tools", "tools",
                                  "benchmark_tool", "benchmark_app.py")
    mo_file = os.path.join(FLAGS.openvino_path, "deployment_tools", "model_optimizer",
                           "mo.py")

    #flags = tf.app.flags
    #mo_file = os.path.join("/", "opt", "intel", "openvino",
    #"deployment_tools", "model_optimizer", "mo.py")
    #bench_app_file = os.path.join("/","opt","intel", "openvino",
    #"deployment_tools", "tools", "benchmark_tool", "benchmark_app.py")

    # check if necessary files exists
    if not os.path.isfile(mo_file) or not os.path.isfile(bench_app_file):
        sys.exit("Openvino not installed!")

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    #FIXME: Conversion not adapted for Windows use yet
    print("\n**********Movidius FP16 conversion**********")
    xml_path = ""
    model_name = ""

    if not os.path.isfile(FLAGS.pb):
        if os.path.isfile(FLAGS.xml):
            xml_path = FLAGS.xml
            print("Already using a converted model -> skipping conversion")
        else:
            sys.exit("Please enter a valid IR! Provided value: " + FLAGS.pb)
    else:
        print("Convert input model to OpenVino format")
        xml_path = convert_model(mo_file, xml_path)

    perform_inference(bench_app_file, is_linux, xml_path)

    print("**********DONE**********")

if __name__ == "__main__":
    app.run(main)

