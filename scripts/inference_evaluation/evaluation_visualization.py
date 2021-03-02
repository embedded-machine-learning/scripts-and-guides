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
from adjustText import adjust_text

# Own modules
import image_utils as im

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

def visualize_latency(df, output_dir):
    # Extract latency data for each network
    #unique_networks = df['Network'].unique()
    #unique_devices = df['Hardware'].unique()
    values = list()
    labels = list()
    ticks = list()
    i = 1

    for index, row in df.iterrows():
        network = row['Model_Short']
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

    im.show_save_figure(fig7, output_dir, 'latency_boxplot', show_image=True)

    # Create the boxplot
    fig1, ax1 = plt.subplots(figsize=(7,12))
    bp = ax1.violinplot(values, showmedians=True, showextrema=True)
    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel("Latency [ms]")
    plt.xlabel("Networks and platforms")
    plt.tight_layout()

    im.show_save_figure(fig1, output_dir, 'latency_violinplot', show_image=True)

def visualize_performance(df, output_dir):
    # Extract latency data for each network
    #unique_networks = df['Network'].unique()
    #unique_devices = df['Hardware'].unique()
    values = list()
    labels = list()
    ticks = list()
    i = 0

    for index, row in df.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        #col = ast.literal_eval(row['DetectionBoxes_Precision/mAP'])
        values.append(row['DetectionBoxes_Precision/mAP'])
        # Add labels
        labels.append(network + "_" + device)
        ticks.append(i)
        i = i + 1

    # Visulization
    # mAP Visualization
    fig1, ax1 = plt.subplots(figsize=(7, 12))
    ax1.set_title('Performance mAP')
    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel('DetectionBoxes_Precision/mAP')
    plt.xlabel("Platforms and network sizes")
    ax1.bar(labels, values)
    plt.tight_layout()

    im.show_save_figure(fig1, output_dir, 'mAP_barplot', show_image=True)

    # Recall Visualization
    values = list()
    labels = list()
    ticks = list()
    i = 0

    for index, row in df.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        # col = ast.literal_eval(row['DetectionBoxes_Precision/mAP'])
        values.append(row['DetectionBoxes_Recall/AR@1'])
        # Add labels
        labels.append(network + "_" + device)
        ticks.append(i)
        i = i + 1

    fig1, ax1 = plt.subplots(figsize=(7, 12))
    ax1.set_title('Performance Recall')
    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel('DetectionBoxes_Recall/AR@1')
    plt.xlabel("Platforms and network sizes")
    ax1.bar(labels, values)
    plt.tight_layout()

    im.show_save_figure(fig1, output_dir, 'Recall_barplot', show_image=True)

def visualize_performance_recall_optimum(latency, performance, output_dir):
    '''
    Find the pareto optimum for models for mAP and latency

    '''

    latency_reduced = latency.drop(columns=['Date']).set_index(['Model_Short', 'Hardware'])
    latency_reduced = latency_reduced[~latency_reduced.index.duplicated(keep='first')]
    performance_reduced = performance.drop(columns=['Date']).set_index(['Model_Short', 'Hardware'])
    performance_reduced = performance_reduced[~performance_reduced.index.duplicated(keep='first')]

    lat_perf_df = pd.merge(latency_reduced, performance_reduced, how='inner', left_index=True, right_index=True)

    print("Available columns: ", lat_perf_df.columns)

    plot_performance_latency(lat_perf_df, output_dir, title='mAP_vs_Latency', y_col='DetectionBoxes_Precision/mAP') #AP at IoU=.50:.05:.95 (primary challenge metric)
    plot_performance_latency(lat_perf_df, output_dir, title='Recall_vs_Latency', y_col='DetectionBoxes_Recall/AR@100') #100 Detections/image
    print("Visualization complete")


def plot_performance_latency(lat_perf_df, output_dir=None, title='mAP_vs_Latency', y_col='DetectionBoxes_Precision/mAP'):
    # Get unique hardware
    hardware_types = list(lat_perf_df.reset_index()['Hardware'].unique())

    performance_max = np.max(lat_perf_df[y_col].values)
    latency_max = np.max(lat_perf_df['Mean_Latency'].values * 1000)

    fig, ax = plt.subplots(figsize=[10, 8])
    plt.title(title)
    plt.xlabel('Latency [ms]')
    plt.ylabel(y_col)
    plt.ylim([0, 1])
    plt.xlim([0, latency_max*1.05])
    plt.grid()
    latency_col = []
    performance_col = []
    texts = []
    for hw in hardware_types:
        hw_type_df = lat_perf_df.reset_index()[lat_perf_df.reset_index()['Hardware'] == hw]
        latency_col.extend(hw_type_df['Mean_Latency'].values * 1000)
        performance_col.extend(hw_type_df[y_col].values)
        ax.scatter(hw_type_df['Mean_Latency'].values * 1000, hw_type_df[y_col].values,
                   label=hw)

        texts.extend([plt.text(hw_type_df['Mean_Latency'].values[i] * 1000,
                               hw_type_df[y_col].values[i],
                               hw_type_df.iloc[i]['Model_Short'])
                      for i in range(hw_type_df.shape[0])])
    plt.legend()

    plt.vlines(20, 0, 1.0, color='red')
    texts.append(plt.text(22, 0.5, "Requirement 20ms", rotation=90, verticalalignment='center'))

    plt.hlines(performance_max, 0, latency_max*1.05, color='red')
    texts.append(plt.text(latency_max/4, performance_max*1.05, "Baseline {:.2f}".format(performance_max), rotation=0, horizontalalignment='center'))

    iter = adjust_text(texts, latency_col, performance_col,
                       arrowprops=dict(arrowstyle="->", color='r', lw=0.5),
                       save_steps=False,
                       ax=ax,
                       # precision=0.001,
                       # expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                       # force_text=(0.01, 0.25), force_points=(0.01, 0.25),
                       )

    im.show_save_figure(fig, output_dir, title, show_image=True)


def evaluate(latency_file, performance_file, output_dir):
    '''

    '''

    # Read all latency files from that folder
    latency = pd.read_csv(latency_file, sep=';')
    visualize_latency(latency, output_dir)

    # Read all performance files
    performance = pd.read_csv(performance_file, sep=';')
    visualize_performance(performance, output_dir)

    # mAP/latency curve
    visualize_performance_recall_optimum(latency, performance, output_dir)



if __name__ == "__main__":
    evaluate(args.latency_file, args.performance_file, args.output_dir)
    # visualize_values(args.infile)

    print("=== Program end ===")
