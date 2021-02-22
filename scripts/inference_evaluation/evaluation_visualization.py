#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize performance and latencies from networks.

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
# ==============================================================================

# The following script uses several method fragments the following script:
# Source:

"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import argparse
import os
import glob
import ast

# Libs
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(
    description='Evaluation and Visualization')
parser.add_argument("-lat", '--latency_file', default=None,
                    help='Directory with latency inputs', required=False)
parser.add_argument("-per", '--performance_file', default=None,
                    help='Directory with performance inputs', required=False)
parser.add_argument("-out", '--output_dir', default='results',
                    help='Output dir', required=False)
args = parser.parse_args()


def visualize_values(infile):
    ''' Load picklefile with report data and visualize it as network on device and with boxplots

    :param infile: Inputfile
    :return: Nothing
    '''

    # Load file
    # open a file, where you stored the pickled data
    file = open(os.path.abspath(infile.strip()), 'rb')
    # dump information to that file
    database = pickle.load(file)
    # close the file
    file.close()
    print("Loaded dataframe: ", database)
    subset = database[['-m', '-d', 'latency (ms)']]
    subset.rename(columns={'latency (ms)': 'latency', '-m': 'network', '-d': 'device'}, inplace=True)
    # subset['latency'] = subset['latency (ms)'].astype(np.float)
    subset = subset.astype({"latency": np.float})
    # subset = subset[['-m', 'latency']]

    # Extract latency data for each network
    unique_networks = subset['network'].unique()
    unique_devices = subset['device'].unique()
    values = list()
    labels = list()
    for network in unique_networks:
        for device in unique_devices:
            col = subset[(subset['network'] == network) & (subset['device'] == device)]['latency'].values
            # col.astype(np.float)
            # col.shape = (-1,1)
            values.append(col)
            # Add labels
            labels.append(network + "_" + device)

    # Create visualization
    # data1 = values[0]
    # data2 = values[1]
    # data = [data1, data2]

    green_diamond = dict(markerfacecolor='g', marker='D')
    fig7, ax7 = plt.subplots()
    ax7.set_title('MobileNetV1 SSD Latencies')
    ax7.boxplot(values, notch=True, flierprops=green_diamond)
    plt.xticks([1, 2, 3, 4], ['CPU_416x416', 'NCS2_416x416', 'CPU_640x416', 'NCS2_640x416'])
    plt.ylabel("Latency [ms]")
    plt.xlabel("Platforms and network sizes")

    plt.axhline(y=20, color='r', linestyle='-')

    ax7.grid(axis='y')
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    # ax7.text(1, 100, "Test")
    anchored_text = AnchoredText("tests: 1000\niterations: 1", loc=2)
    ax7.add_artist(anchored_text)

    plt.show()

    # Create the boxplot
    fig1, ax1 = plt.subplots()
    bp = ax1.violinplot(values, showmedians=True, showextrema=True)
    plt.xticks([1, 2], ['MobileNetV1SSD_CPU_416x416', 'MobileNetV1SSD_CPU_640x416'])
    plt.ylabel("Latency [ms]")
    plt.xlabel("Networks and platforms")
    plt.show()


# def read_files_to_df(dir: str, ext: str = 'csv') -> pd.DataFrame:
#     csv_list = glob.glob(os.path.join(dir, '*.' + ext))
#
#     df = pd.DataFrame()
#     for file in csv_list:
#         csvdf = pd.read_csv(file)
#
#         df.join(csvdf)
#
#     return df

def visualize_latency(df, output_dir):
    # Extract latency data for each network
    unique_networks = df['Network'].unique()
    unique_devices = df['Hardware'].unique()
    values = list()
    labels = list()
    ticks = list()
    i = 1

    for index, row in df.iterrows():
        network = row['Network']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        col = ast.literal_eval(row['Latencies'])
        #col = ast.literal_eval(df[(df['Network'] == network) & (df['Hardware'] == device)]['Latencies'][0])
        # col = df[(df['network'] == network) & (df['hardware'] == device)]['latency'].values
        # col.astype(np.float)
        # col.shape = (-1,1)
        values.append(np.array(col) * 1000)
        # Add labels
        labels.append(network + "_" + device)
        ticks.append(i)
        i = i + 1

    # for network in unique_networks:
    #     for device in unique_devices:
    #         print("Processing {} on {}".format(network, device))
    #         col = ast.literal_eval(df[(df['Network'] == network) & (df['Hardware'] == device)]['Latencies'][0])
    #         # col = df[(df['network'] == network) & (df['hardware'] == device)]['latency'].values
    #         # col.astype(np.float)
    #         # col.shape = (-1,1)
    #         values.append(np.array(col)*1000)
    #         # Add labels
    #         labels.append(network + "_" + device)
    #         ticks.append(i)
    #         i = i + 1

    # Visualization
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig7, ax7 = plt.subplots(figsize=(7,12))
    ax7.set_title('Latencies')
    ax7.boxplot(values, notch=True, flierprops=green_diamond)
    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel("Latency [ms]")
    plt.xlabel("Platforms and network sizes")

    plt.axhline(y=20, color='r', linestyle='-')

    ax7.grid(axis='y')
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    # ax7.text(1, 100, "Test")
    anchored_text = AnchoredText("Inferences: {}".format(values[0].shape[0]), loc=2)
    ax7.add_artist(anchored_text)
    plt.tight_layout()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'latency_boxplot'))
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    # Create the boxplot
    fig1, ax1 = plt.subplots(figsize=(7,12))
    bp = ax1.violinplot(values, showmedians=True, showextrema=True)
    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel("Latency [ms]")
    plt.xlabel("Networks and platforms")
    plt.tight_layout()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'latency_violinplot'))
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def evaluate(latency_file, performance_file, output_dir):
    '''

    '''

    # Read all latency files from that folder
    latency = pd.read_csv(latency_file, sep=';')
    visualize_latency(latency, output_dir)

    # Read all performance files
    performance = pd.read_csv(performance_file, sep=';')


if __name__ == "__main__":
    evaluate(args.latency_file, args.performance_file, args.output_dir)
    # visualize_values(args.infile)

    print("=== Program end ===")
