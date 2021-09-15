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
import os, sys, csv, argparse
from datetime import datetime

# Libs
import argparse

# Own modules
#sys.path.append('../../inference_evaluation')
#import inference_utils as util


__author__ = 'Matvey Ivanov'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['Matvey Ivanov']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Matvey Ivanov'
__email__ = 'matvey.ivanov@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='NCS2 settings test')
parser.add_argument("-ar", '--avg_rep', type=str,
                    help='report containing average counters from the Openvino benchmark app', required=False)
parser.add_argument("-ir", '--inf_rep', type=str,
                    help='report containing execution information from the Openvino benchmark app', required=False)
parser.add_argument("-hw", '--hardware_name', default=None, type=str,
                    help='Hardware name for collecting statistical data in the reports.', required=False)
parser.add_argument("-l", '--delimiter', type=str, default=";",
                    help='whatever the csv uses as delimiter', required=False)
parser.add_argument("-o", '--output_path', type=str, default="results/latency.csv",
                    help='Output file to write new or to apped to.', required=False)
parser.add_argument('-id', '--index_save_file', type=str, default='./tmp/index.txt',
                    help='Path to put index file to keep the same key for different types of measurements.',
                    required=False)

parser.add_argument('--save_new', dest='save_new', action='store_true')
# parser.add_argument('--append', dest='save_new', action='store_false')
parser.set_defaults(save=False)
args = parser.parse_args()
print(args)

# keywords used in the Openvino reports
# extract from info rep: name, hardware, batch, mode (sync,async), throughput, latency
keywords = ["target device", "--path_to_model", "number of parallel infer requests",
            "API", "batch size", "latency (ms)", "throughput"]  # , "precision"]
# keywords used in the EML data structure
latency_keywords = ["Index", "Date", "Model", "Model_Short", "Framework", "Network", "Resolution", "Dataset",
                    "Custom_Parameters",
                    "Hardware", "Hardware_Optimization", "Batch_Size", "Throughput", "Mean_Latency", "Latencies"]


def file_exists(file_path):
    """Simple function checks if file exists at file_path

    :param file_path: path to file to check
    :return: True if file exists, False if it doesn't
    """
    if not os.path.isfile(file_path):
        print("Could not find file under:", file_path)
        return False
    return True


def read_csv_report(datafile_path, delimiter):
    """Parses data from csv file to list and returns it

    :param datafile_path: path to report file
    :param delimiter: delimiter used in the report file
    :return: list with data from csv or None when unable to parse data
    """
    with open(datafile_path, "r", newline='') as report_file:
        if delimiter == None:
            delimiter = ";"
        print("using ", delimiter, " as delimiter")
        return list(csv.reader(report_file, delimiter=delimiter, quotechar='|'))
    return None


def extract_information_avg_rep(report_data):
    # structure of data: ['layerName', 'execStatus', 'layerType', 'execType', 'realTime (ms)', 'cpuTime (ms)']

    # go over the data, cast it to float at append to a list
    layers_durations = []
    layer_names = []
    network_duration = 0
    for row in report_data:
        if row == []:
            break  # if line is empty, end loop
        # print(row, len(row))
        # check if the rows contains valid data in format xxx.xxx where the . signifies a float value
        if len(row) > 3 and "." in row[4]:
            if float(row[4]) > 0:
                # print(row[0], float(row[4]))
                layer_names.append(row[0])
                layers_durations.append(float(row[4]))
            if row[0] == "Total":
                network_duration = float(row[4])
    # print("\nnetwork duration: {0:5f} ms\n".format(network_duration))

    return layers_durations, layer_names, network_duration

def generate_measurement_index(model_name):
    '''
    Generate an index for a measurement that is used as a database key.

    :param model_name: Model name long
    :return: index
    '''
    index = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_name
    return index

def extract_information_inf_rep(report_data):
    # inf rep data format is ["Command line parameters", parameters]
    extracted_inf = {}  # prepare a dictionary where to put the extracted information into

    # FIXME: How to handle precisions that are not connected to NCS2?
    extracted_inf["precision"] = "UNSPECIFIED"
    extracted_inf["custom_params"] = []

    for row in report_data:
        if row == []:
            continue  # if line is empty, skip row

        if len(row) > 0 and row[0] in keywords:
            print(row)
            try:
                if row[0] == "--path_to_model":
                    # augment the data - more information can be extracted from model file path
                    # e.g. /home/matvey/projects/models/xilinx_models/xilinx_model_movid/tf2oda_ssdmobilenetv2_300x300_pedestrian_D100_LR08.csv
                    extracted_inf["--path_to_model"] = row[1]
                    if "/" in extracted_inf["--path_to_model"]:
                        if ".xml" in extracted_inf["--path_to_model"]:
                            extracted_inf["full_name"] = extracted_inf["--path_to_model"].split("/")[-2]  # FIX AW: Model name is always the folder name
                            # extracted_inf["full_name"] = extracted_inf["--path_to_model"].split("/")[-1].split(".xml")[0] # get full name from path to model

                            #model_info = util.get_info_from_modelname(extracted_inf["full_name"])
                            #model_info['model_name'],
                            #model_info['model_short_name'],
                            #model_info['framework'],
                            #model_info['network'],
                            #str(model_info['resolution']),
                            #model_info['dataset'],
                            #str(model_info['custom_parameters']),
                            #hardware_name,
                            #str(model_info['hardware_optimization']


                            if "_" in extracted_inf["full_name"]:
                                try:
                                    info = get_info_from_modelname(extracted_inf["full_name"], model_short_name=None, model_optimizer_prefix=['TRT', 'OV'])

                                    extracted_inf["short_name"] = info['model_short_name'] #extracted_inf["full_name"]   #is the same as the full name
                                    extracted_inf["resolution"] = info['resolution'] #list(map(int, (str(extracted_inf["full_name"]).split('_')[2]).split('x')))
                                    extracted_inf["framework"] = info['framework'] #extracted_inf["full_name"].split("_")[0] #extracted_inf["full_name"].split("_")[0]
                                    extracted_inf["network"] = info['network']
                                    #extracted_inf["resolution"] = extracted_inf["full_name"].split("_")[2]
                                    extracted_inf["dataset"] = info['dataset'] #extracted_inf["full_name"].split("_")[3]
                                    extracted_inf["hwoptimization"] = info['hardware_optimization']
                                    # extracted_inf["custom_params"] = extracted_inf["full_name"].split("_")[4:]

                                    custom_list = []
                                    model_optimizer_prefix = "OV"
                                    if len(extracted_inf["full_name"].split("_", 4)) > 4:
                                        rest_parameters = extracted_inf["full_name"].split("_", 4)[4]

                                        for r in rest_parameters.split("_"):
                                            if str(r).startswith(model_optimizer_prefix):
                                                extracted_inf["hwoptimization"] = r
                                            else:
                                                custom_list.append(r)
                                    #extracted_inf['custom_params'] = str(custom_list)
                                    extracted_inf['custom_params'] = custom_list
                                except:
                                    print("Could not split ", extracted_inf["full_name"],
                                          " to extract data, because a '_' is missing. " \
                                          "is the format correct?")
                            else:
                                extracted_inf["short_name"] = None
                                extracted_inf["framework"] = None
                                extracted_inf["resolution"] = None
                                extracted_inf["dataset"] = None
                                extracted_inf["custom_params"] = None
                        else:
                            extracted_inf["full_name"] = None
                    else:
                        print("No ’/’ in ", extracted_inf["full_name"],
                              ". Please check if parsed information is correct.")
                    continue

                # FIXME: This is a special case, we need to make it general
                if row[0] == "target device" and row[1] == "MYRIAD":
                    extracted_inf["precision"] = "FP16"

                extracted_inf[row[0]] = row[1]  # add data to dict
                print("filling in", row[1])
            except:
                continue  # if row[0] not in keywords, skip row

    # Add OpenVino HW optimization
    # extracted_inf["hwoptimization"] = "OV_" + extracted_inf["precision"]

    return extracted_inf

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

def reformat_inf(extracted_inf, hardware_name=None):
    # build a dataframe according to the latency data format in latency_keywords
    index = generate_measurement_index(extracted_inf["full_name"])

    # Save index
    # Save index to a file
    os.makedirs(os.path.dirname(args.index_save_file), exist_ok=True)

    file1 = open(args.index_save_file, 'w')
    file1.write(index)
    print("Index {} used for latency measurement".format(index))

    new_frame = [index]
    new_frame.append(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))  # Date
    new_frame.append(extracted_inf["full_name"])  # Model
    new_frame.append(extracted_inf["short_name"])  # Model_Short
    new_frame.append(extracted_inf["framework"])  # Framework
    new_frame.append(extracted_inf["network"])  # Network
    new_frame.append(extracted_inf["resolution"])  # Resolution
    new_frame.append(extracted_inf["dataset"])  # Dataset

    custom_complete = list()
    custom_complete.append(extracted_inf["number of parallel infer requests"])
    custom_complete.append(extracted_inf["API"])
    custom_complete.append(extracted_inf["precision"])
    custom_complete.extend(extracted_inf["custom_params"])

    new_frame.append(custom_complete)  # Custom_Parameters
    if hardware_name:
        new_frame.append(hardware_name + "_" + extracted_inf["target device"])  # Hardware_type including hardware device
    else:
        new_frame.append(extracted_inf["target device"])  # Hardware_type only
    new_frame.append(extracted_inf["hwoptimization"])  # Hardware_Optimization
    new_frame.append(extracted_inf["batch size"])  # Batch_Size
    new_frame.append(extracted_inf["throughput"])  # Throughput
    new_frame.append(extracted_inf["latency (ms)"])  # Mean_Latency
    new_frame.append(None)  # Latencies
    print(new_frame)
    return new_frame

def parse_avg_report(report_path, delimiter):
    # check if passed report file exists
    if not file_exists(report_path):
        return

    # get the data from file assuming an Openvino average counters benchmark report
    report_data = read_csv_report(report_path, delimiter)
    layers_durations, layer_names, network_duration = extract_information_avg_rep(report_data)

    return layers_durations, layer_names, network_duration


def parse_inf_report(report_path, delimiter):
    # check if passed report file exists
    if not file_exists(report_path):
        return

    # get the data from file assuming an  openvino benchmark app information report

    report_data = read_csv_report(report_path, delimiter)
    extracted_inf = extract_information_inf_rep(report_data)

    return extracted_inf


def save_file(latency_keywords, reformated_inf, output_path, save_new):
    if not file_exists(output_path) or save_new:
        with open(output_path, "w", newline='') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(latency_keywords)  # write the upper column according to dataframe
            writer.writerow(reformated_inf)

            print("Writing file {} successful!".format(output_path))
    else:
        with open(output_path, "a+", newline='') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(reformated_inf)

            print("Writing to " + output_path + " successful!")


# def save_new_rep(reformated_inf, output_path):
#     #print(extracted_inf)
#     # create a new file handle for new report
#     with open(output_path, "w") as f:
#         writer = csv.writer(f, delimiter=";")
#         writer.writerow(latency_keywords) # write the upper column according to dataframe
#         writer.writerow(reformated_inf)
#
#         print("Writing file {} successful!".format(output_path))
#
#
# def append_to_file(reformated_inf, existing_file):
#     if not file_exists(existing_file):
#         print("Parsed file on which to append data ", existing_file, " does not exist.")
#         return
#     # open existing file and append extracted information
#     with open(existing_file, "a+") as f:
#         writer = csv.writer(f, delimiter=";")
#         writer.writerow(reformated_inf)
#
#         print("Writing to " + existing_file + " successful!")

if __name__ == "__main__":

    if args.avg_rep:
        print("Parsing the average counters report...")
        layers_durations, layer_names, network_duration = parse_avg_report(args.avg_rep, args.delimiter)
        print("\nlayers_durations\n", layers_durations, "\nlayer_names\n", layer_names, "\nnetwork_duration\n",
              network_duration)
    else:
        print("Skipping the average counters report...")

    if args.inf_rep:
        print("Parsing the inference information report...")
        extracted_inf = parse_inf_report(args.inf_rep, args.delimiter)
        print("\nextracted_inf\n", extracted_inf)
        reformated_inf = reformat_inf(extracted_inf, args.hardware_name)
        print("\nreformated_inf\n", reformated_inf)
    else:
        print("Skipping the inference information report...")

    save_file(latency_keywords, reformated_inf, args.output_path, args.save_new)

    # if args.save_new:
    #     print("Saving extracted information to new file...")
    #     save_new_rep(reformated_inf, args.output_path)
    # else:
    #     # append data to existing file
    #     append_to_file(reformated_inf, args.output_path)
