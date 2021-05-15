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
import warnings

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
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(
    description='Evaluation and Visualization')
parser.add_argument("-lat", '--latency_file', default=None,
                    help='Directory with latency inputs', required=False)
parser.add_argument("-per", '--performance_file', default=None,
                    help='Directory with performance inputs', required=False)
parser.add_argument("-in", '--input_combined_file', default=None,
                    help='Combined input file', required=False)
parser.add_argument("-out", '--output_dir', default='results',
                    help='Output dir', required=False)
parser.add_argument("-hwoptref", '--hwopt_reference', default=None,
                    help='This is the name of the hwoptimization parameter that is used a reference'
                         'for latency and performance measurements. Default value is None', required=False)
args = parser.parse_args()


def visualize_latency(df, output_dir):
    '''
    Visualize latency in different plots. Each hardware has a separate colour.


    '''
    # Plot latency for all networks and hardware
    # unique_networks = df['Network'].unique()

    df_perf = df.sort_values(by=['Mean_Latency'], ascending=True)

    values = list()
    labels = list()
    for index, row in df_perf.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        hwopt = row['Hardware_Optimization']
        print("Processing {} on {}".format(network, device))
        if not np.isnan(row['Latencies']):
            col = ast.literal_eval(row['Latencies'])
            values.append(np.array(col))
        else:
            warnings.warn("No single latencies available for " + row['Model'] + ". Use mean latency for the graphs.")
            mean_array = np.array(row['Mean_Latency'])
            mean_array = mean_array.reshape(-1)
            values.append(mean_array)
        labels.append(network + "_" + device + "_" + hwopt)

    plot_boxplot(values, labels, output_dir, title="Latency All Hardware")
    plot_violin_plot(values, labels, output_dir, title="Latency All Hardware")

    # Plot latencies per hardware
    unique_hardware = df_perf['Hardware'].unique()
    for hw in unique_hardware:
        sub_df = df_perf[df_perf['Hardware'] == hw]
        values = list()
        labels = list()
        for index, row in sub_df.iterrows():
            network = row['Model_Short']
            device = row['Hardware']
            hwopt = row['Hardware_Optimization']
            print("Processing {} on {}".format(network, hw))

            if not np.isnan(row['Latencies']):
                col = ast.literal_eval(row['Latencies'])
                values.append(np.array(col))
            else:
                warnings.warn(
                    "No single latencies available for " + row['Model'] + ". Use mean latency for the graphs.")
                mean_array = np.array(row['Mean_Latency'])
                mean_array = mean_array.reshape(-1)
                values.append(mean_array)
            labels.append(network + "_" + hwopt)
            # col = ast.literal_eval(row['Latencies'])
            # values.append(np.array(col) * 1000)
            # labels.append(network)

        plot_boxplot(values, labels, output_dir, title="Latency " + hw)
        plot_violin_plot(values, labels, output_dir, title="Latency " + hw)


def plot_violin_plot(values, labels, output_dir, title='Latency', xlabel="Models and platforms", ylabel="Latency [ms]"):
    '''
    Plot violinplot

    :argument
        values: list of 1D-array of values
        labels: List of label names
        output_dir: outputdir, where to save the image

    :return
        None

    '''
    # Create the Violinplot
    fig1, ax1 = plt.subplots(figsize=(5, 8))
    bp = ax1.violinplot(values, showmedians=True, showextrema=True)
    ax1.set_title(title)

    ticks = list(np.linspace(1, len(labels), len(labels)).astype(np.int))

    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    add_range = (np.max(values) - np.min(values)) * 0.10
    plt.ylim([np.min(values) - add_range, np.max(values) + add_range])

    anchored_text = AnchoredText("Inferences: {}".format(values[0].shape[0]), loc=2)
    ax1.add_artist(anchored_text)
    ax1.grid(axis='y')
    plt.tight_layout()
    im.show_save_figure(fig1, output_dir, title.replace(' ', '_') + '_violinplot', show_image=False)


def plot_boxplot(values, labels, output_dir=None, title='Latencies', xlabel="Models and platforms",
                 ylabel="Latency [ms]"):
    '''
    Plot Violinplot

    :argument
        values: list of 1D-array of values
        labels: List of label names
        output_dir: outputdir, where to save the image

    :return
        None
    '''
    # Visualization
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig7, ax7 = plt.subplots(figsize=(5, 8))

    ax7.set_title(title)

    ax7.boxplot(values, notch=True, flierprops=green_diamond)

    ticks = list(np.linspace(1, len(labels), len(labels)).astype(np.int))

    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    add_range = (np.max(values) - np.min(values)) * 0.10
    plt.ylim([np.min(values) - add_range, np.max(values) + add_range])

    # plt.axhline(y=20, color='r', linestyle='-')
    ax7.grid(axis='y', which='both')
    ax7.minorticks_on()
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    # ax7.text(1, 100, "Test")
    anchored_text = AnchoredText("Inferences: {}".format(values[0].shape[0]), loc=2)
    ax7.add_artist(anchored_text)
    plt.tight_layout()
    im.show_save_figure(fig7, output_dir, title.replace(' ', '_') + '_boxplot', show_image=False)


def plot_bar(values, labels, output_dir, title='Performance mAP', xlabel="Models and platforms",
             ylabel='DetectionBoxes_Precision/mAP'):
    # mAP Visualization
    fig1, ax1 = plt.subplots(figsize=(7, 12))
    ax1.set_title(title)
    ax1.grid()

    ticks = list(np.linspace(0, len(labels), len(labels)).astype(np.int))

    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    ax1.bar(labels, values)
    for i, v in enumerate(values):
        ax1.text(i - .5, v * 1.02, "{:.3f}".format(v))
    # for i, v in enumerate(values):
    #    ax1.text(v + 3, i + .25, "{:.2f}".format(v))
    plt.tight_layout()
    im.show_save_figure(fig1, output_dir, title.replace(' ', '_') + '_barplot', show_image=False)


def visualize_performance(df, output_dir):
    '''


    '''

    values = list()
    labels = list()

    df_perf = df.sort_values(by=['DetectionBoxes_Precision/mAP'], ascending=False)

    for index, row in df_perf.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        values.append(row['DetectionBoxes_Precision/mAP'])
        # Add labels
        labels.append(network + " " + device)

    # Visulization
    plot_bar(values, labels, output_dir, title="Mean Average Precision", xlabel="Models and Hardware", ylabel="mAP")

    # Recall Visualization
    df_perf = df.sort_values(by=['DetectionBoxes_Recall/AR@1'], ascending=False)

    values = list()
    labels = list()

    for index, row in df_perf.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        values.append(row['DetectionBoxes_Recall/AR@1'])
        # Add labels
        labels.append(network + " " + device)

    plot_bar(values, labels, output_dir, title="Recall", xlabel="Models and Hardware", ylabel="Recall/AR@1")


def visualize_performance_recall_optimum(latency, performance, output_dir):
    '''
    Find the pareto optimum for models for mAP and latency

    '''

    latency_reduced = latency.drop(columns=['Date']).set_index(['Model_Short', 'Hardware'])
    latency_reduced = latency_reduced[~latency_reduced.index.duplicated(keep='first')]
    performance_reduced = performance.drop(columns=['Date']).set_index(['Model_Short', 'Hardware'])
    performance_reduced = performance_reduced[~performance_reduced.index.duplicated(keep='first')]

    lat_perf_df = pd.merge(latency_reduced, performance_reduced, how='inner', left_index=True, right_index=True)
    lat_perf_df = lat_perf_df.reset_index()

    print("Available columns: ", lat_perf_df.columns)
    # Plot for all hardware
    plot_performance_latency(lat_perf_df, output_dir, title='mAP vs Latency All Hardware',
                             y_col='DetectionBoxes_Precision/mAP',
                             ylim=[0, 1])  # AP at IoU=.50:.05:.95 (primary challenge metric)
    plot_performance_latency(lat_perf_df, output_dir, title='mAP vs Latency Zoom', y_col='DetectionBoxes_Precision/mAP')

    plot_performance_latency(lat_perf_df, output_dir, title='Recall vs Latency',
                             y_col='DetectionBoxes_Recall/AR@100')  # 100 Detections/image

    # Plot for each hardware separately
    unique_hardware = lat_perf_df['Hardware'].unique()
    for hw in unique_hardware:
        sub_df = lat_perf_df[lat_perf_df['Hardware'] == hw]
        plot_performance_latency(sub_df, output_dir, title='mAP vs Latency Zoom ' + hw,
                                 y_col='DetectionBoxes_Precision/mAP',
                                 plot_separation='Network_x')
        plot_performance_latency(sub_df, output_dir, title='Recall vs Latency Zoom ' + hw,
                                 y_col='DetectionBoxes_Precision/mAP',
                                 plot_separation='Network_x')


def plot_performance_latency(lat_perf_df, output_dir=None, title='mAP_vs_Latency', y_col='DetectionBoxes_Precision/mAP',
                             ylim=None, xlim=None, latency_req=20, plot_separation='Hardware'):
    # Get unique hardware
    hardware_types = list(lat_perf_df[plot_separation].unique())

    performance_max = np.max(lat_perf_df[y_col].values)
    performance_min = np.min(lat_perf_df[y_col].values)
    latency_max = np.max(lat_perf_df['Mean_Latency'].values)

    fig, ax = plt.subplots(figsize=[8, 8])
    plt.title(title)
    plt.xlabel('Latency [ms]')
    plt.ylabel(y_col)
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([performance_min * 0.95, performance_max * 1.05])
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, latency_max * 1.05])
    plt.grid()
    latency_col = []
    performance_col = []
    texts = []
    for hw in hardware_types:
        hw_type_df = lat_perf_df[lat_perf_df[plot_separation] == hw]
        latency_col.extend(hw_type_df['Mean_Latency'].values)
        performance_col.extend(hw_type_df[y_col].values)
        ax.scatter(hw_type_df['Mean_Latency'].values, hw_type_df[y_col].values,
                   label=hw)

        texts.extend([plt.text(hw_type_df['Mean_Latency'].values[i],
                               hw_type_df[y_col].values[i],
                               hw_type_df.iloc[i]['Model_Short'])
                      for i in range(hw_type_df.shape[0])])
    plt.legend()

    if latency_req:
        plt.vlines(latency_req, 0, 1.0, color='red')
        texts.append(
            plt.text(latency_req + 2, 0.5, "Requirement {:.2f}ms".format(20), rotation=90, verticalalignment='center'))

    plt.hlines(performance_max, 0, latency_max * 1.05, color='blue', linestyles="dotted")
    texts.append(
        plt.text(latency_max / 4, performance_max * 1.05, "Baseline {:.2f}".format(performance_max), rotation=0,
                 horizontalalignment='center'))

    iter = adjust_text(texts, latency_col, performance_col,
                       arrowprops=dict(arrowstyle="->", color='r', lw=0.5),
                       save_steps=False,
                       ax=ax,
                       # precision=0.001,
                       # expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                       # force_text=(0.01, 0.25), force_points=(0.01, 0.25),
                       )

    im.show_save_figure(fig, output_dir, title.replace(' ', '_') + '_scatter', show_image=False)


def get_performance_deltas_for_hardware_optimizations(performance):
    '''


    '''

    # For each model and hardware
    enhanced_performance = performance.copy()
    enhanced_performance['Relative_mAP'] = 0
    x = enhanced_performance.groupby(['Framework', 'Network', 'Resolution', 'Dataset', 'Custom_Parameters', 'Hardware'])

    for name, group in x:
        print(name)
        print(group)

        if len(group[(pd.isnull(group['Hardware_Optimization'])) |
                     (group['Hardware_Optimization'] == 'None')]) > 0:
            original_performance = \
                group[(pd.isnull(group['Hardware_Optimization'])) |
                      (group['Hardware_Optimization'] == 'None')]['DetectionBoxes_Precision/mAP@.50IOU'].values[0]

            for i, row in group.iterrows():
                current_performance = row['DetectionBoxes_Precision/mAP@.50IOU']
                relative_performance = np.round((current_performance - original_performance) / current_performance, 3)

                enhanced_performance.loc[(enhanced_performance['Framework'] == row['Framework']) &
                                         (enhanced_performance['Network'] == row['Network']) &
                                         (enhanced_performance['Resolution'] == row['Resolution']) &
                                         (enhanced_performance['Dataset'] == row['Dataset']) &
                                         (enhanced_performance['Custom_Parameters'] == row['Custom_Parameters']) &
                                         (enhanced_performance['Hardware'] == row['Hardware']) &
                                         (enhanced_performance['Hardware_Optimization'] == row[
                                             'Hardware_Optimization']),
                                         'Relative_mAP'] = relative_performance
                print("Created value for ", name)
        else:
            warnings.warn("No original latency for " + str(name))

    return enhanced_performance


def get_latency_deltas_for_hardware_optimizations(latency, hwopt_reference=None):
    '''
    Calculate relative latencies

    :argument



    '''

    # For each model and hardware
    enhanced_latency = latency.copy()
    enhanced_latency['Relative_Latency'] = 0
    x = enhanced_latency.groupby(['Framework', 'Network', 'Resolution', 'Dataset', 'Custom_Parameters', 'Hardware'])

    for name, group in x:
        print(name)
        print(group)

        # Look for executions without any hardware optimizations to use to compare to
        if len(group[(pd.isnull(group['Hardware_Optimization'])) |
                     (group['Hardware_Optimization'] == hwopt_reference)]) > 0:
            original_latency = \
                group[(pd.isnull(group['Hardware_Optimization'])) |
                      (group['Hardware_Optimization'] == hwopt_reference)]['Mean_Latency'].values[0]

            for i, row in group.iterrows():
                current_latency = row['Mean_Latency']
                relative_latency = np.round((current_latency - original_latency) / current_latency, 3)

                enhanced_latency.loc[(enhanced_latency['Framework'] == row['Framework']) &
                                     (enhanced_latency['Network'] == row['Network']) &
                                     (enhanced_latency['Resolution'] == row['Resolution']) &
                                     (enhanced_latency['Dataset'] == row['Dataset']) &
                                     (enhanced_latency['Custom_Parameters'] == row['Custom_Parameters']) &
                                     (enhanced_latency['Hardware'] == row['Hardware']) &
                                     (enhanced_latency['Hardware_Optimization'] == row['Hardware_Optimization']),
                                     'Relative_Latency'] = relative_latency
                print("Created value for ", name)
        else:
            warnings.warn("No original latency for " + str(name))

    return enhanced_latency


def visualize_relative_latencies(latencies, output_dir, measurement='Relative_Latency',
                                 title="Latency Delta ",
                                 xlabel="Model Optimization Method",
                                 ylabel="Relative Latency"
                                 ):
    '''
    Visualize relative latencies

    '''

    # Plot latency for all networks and hardware
    # unique_networks = df['Network'].unique()
    for hw_name, hw_group in latencies.groupby('Hardware'):
        values = list()
        labels = list()
        for name, group in hw_group.groupby('Hardware_Optimization'):
            # group['Relative_Latency']
            values.append(np.array(group[measurement]))
            labels.append(name)

        # if len(values[0])>1:
        plot_boxplot(values, labels, output_dir, title=title + " for " + hw_name, xlabel=xlabel,
                     ylabel=ylabel)
        plot_violin_plot(values, labels, output_dir, title=title + " for " + hw_name, xlabel=xlabel,
                         ylabel=ylabel)
        # else:
        #    first_values = []
        #    for e in values:
        #        first_values.append(e[0])
        #    plot_bar(first_values, labels, output_dir, title=title + " for " + hw_name, xlabel=xlabel,
        #             ylabel=ylabel)


def evaluate(latency_file, performance_file, output_dir, hwopt_reference=None, input_combined_file=None):
    '''

    '''

    if input_combined_file:
        if not os.path.exists(input_combined_file):
            raise Exception(input_combined_file + " does not exist. Quit program")
        combined = pd.read_csv(input_combined_file, sep=';')
        print("Loaded combined latency and performance file: ", input_combined_file)
        latency = combined
        performance = combined

    else:
        # Read all latency files from that folder
        if not os.path.exists(latency_file):
            raise Exception(latency_file + " does not exist. Quit program")
        latency = pd.read_csv(latency_file, sep=';')
        # Read all performance files
        if not os.path.exists(performance_file):
            warnings.warn(performance_file + " does not exist. Continue with functions that only support latency.")
            performance = None
        else:
            performance = pd.read_csv(performance_file, sep=';')

    relative_latencies = get_latency_deltas_for_hardware_optimizations(latency, hwopt_reference)
    visualize_relative_latencies(relative_latencies, output_dir)

    # Visualize latency
    visualize_latency(latency, output_dir)

    if performance:
        # Visualize relative performance
        relative_performance = get_performance_deltas_for_hardware_optimizations(performance)
        visualize_relative_latencies(relative_performance, output_dir, measurement='Relative_mAP',
                                     title="Performance Delta for Hardware Optimization Method",
                                     ylabel="Relative Performance"
                                     )
        # Visualize Performance
        visualize_performance(performance, output_dir)

        # mAP/latency curve
        visualize_performance_recall_optimum(latency, performance, output_dir)


if __name__ == "__main__":
    evaluate(args.latency_file, args.performance_file, args.output_dir, args.hwopt_reference, args.input_combined_file)
    # visualize_values(args.infile)

    print("=== Program end ===")
