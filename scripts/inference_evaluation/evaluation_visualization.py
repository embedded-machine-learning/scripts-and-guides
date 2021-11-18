#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize performance and latencies from networks.

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
parser.add_argument("-latreq", '--latency_requirement', default=None,
                    help='Latency requirement in ms. Default is None and then no line is drawn.', required=False)
parser.add_argument("-perfreq", '--performance_requirement', default=None,
                    help='Performance requirement', required=False)
parser.add_argument("-hwoptref", '--hwopt_reference', default=None,
                    help='This is the name of the hwoptimization parameter that is used a reference'
                         'for latency and performance measurements. Default value is None', required=False)

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
parser.add_argument("-latreq", '--latency_requirement', default=None,
                    help='Latency requirement in ms. Default is None and then no line is drawn.', required=False)
parser.add_argument("-perfreq", '--performance_requirement', default=None,
                    help='Performance requirement', required=False)
parser.add_argument("-hwoptref", '--hwopt_reference', default=None,
                    help='This is the name of the hwoptimization parameter that is used a reference'
                         'for latency and performance measurements. Default value is None', required=False)
parser.add_argument("-b", '--plot_baseline', help="Plot a baseline in the graph from the best performing network.",
                    action='store_true', default=False)
args = parser.parse_args()
print(args)


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
        # FIXME: This is not a clean way to check if the field is empty

        if not row['Latencies'] is None:  # or not np.isnan(row['Latencies']):
            col = ast.literal_eval(row['Latencies'])
            values.append(np.array(col))
        else:
            warnings.warn("No single latencies available for " + row['Model'] + ". Use mean latency for the graphs.")
            mean_array = np.array(row['Mean_Latency'])
            mean_array = mean_array.reshape(-1)
            values.append(mean_array)
        labels.append(str(network) + "_" + str(device) + "_" + str(hwopt))

    max_val = np.max(max(values, key=tuple))
    min_val = np.min(min(values, key=tuple))
    plot_boxplot(values, labels, output_dir, title="Latency All Hardware", max_val=max_val, min_val=min_val)
    plot_violin_plot(values, labels, output_dir, title="Latency All Hardware", max_val=max_val, min_val=min_val)

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

            if not row['Latencies'] is None:  # and not np.isnan(row['Latencies']):
                col = ast.literal_eval(row['Latencies'])
                values.append(np.array(col))
            else:
                warnings.warn(
                    "No single latencies available for " + row['Model'] + ". Use mean latency for the graphs.")
                mean_array = np.array(row['Mean_Latency'])
                mean_array = mean_array.reshape(-1)
                values.append(mean_array)
            labels.append(str(network) + "_" + str(hwopt))
            # col = ast.literal_eval(row['Latencies'])
            # values.append(np.array(col) * 1000)
            # labels.append(network)

        max_val = np.max(max(values, key=tuple))
        min_val = np.min(min(values, key=tuple))

        plot_boxplot(values, labels, output_dir, title="Latency " + hw, max_val=max_val, min_val=min_val)
        plot_violin_plot(values, labels, output_dir, title="Latency " + hw, max_val=max_val, min_val=min_val)


def plot_violin_plot(values, labels, output_dir, title='Latency', xlabel="Models and platforms", ylabel="Latency [ms]",
                     max_val=1, min_val=0):
    '''
    Plot violinplot

    :argument
        values: list of 1D-array of values
        labels: List of label names
        output_dir: outputdir, where to save the image

    :return
        None

    '''
    # Extract important values
    df_table = pd.DataFrame(columns=['min', 'max', 'q25', 'median', 'q75', 'mean'])
    for v in zip(labels, values):
        df_table = df_table.append(pd.Series(name=v[0], data=[np.min(v[1]), np.max(v[1]), np.quantile(v[1], 0.25),
                                                              np.quantile(v[1], 0.50), np.quantile(v[1], 0.75),
                                                              np.mean(v[1])],
                                             index=['min', 'max', 'q25', 'median', 'q75', 'mean']))

    df_table.index.name = 'Model'
    df_table.round(2).astype(float).to_csv(os.path.join(output_dir, title.replace(' ', '_') + '_violinplot' + '.csv'),
                                           sep=";")

    # Create the Violinplot
    fig1, ax1 = plt.subplots(figsize=(5, 8))
    bp = ax1.violinplot(values, showmedians=True, showextrema=True)
    ax1.set_title(title)

    ticks = list(np.linspace(1, len(labels), len(labels)).astype(int))

    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    # max_val = np.array(values).max() #np.max(max(values, key=tuple))
    # min_val = np.array(values).min() #np.min(min(values, key=tuple))
    add_range = (max_val - min_val) * 0.10
    plt.ylim([min_val - add_range, max_val + add_range])

    anchored_text = AnchoredText("Inferences: {}".format(values[0].shape[0]), loc=2)
    ax1.add_artist(anchored_text)
    ax1.grid(axis='y')
    plt.tight_layout()

    im.show_save_figure(fig1, output_dir, title.replace(' ', '_') + '_violinplot', show_image=False)


def plot_boxplot(values, labels, output_dir=None, title='Latencies', xlabel="Models and platforms",
                 ylabel="Latency [ms]", max_val=1, min_val=0):
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

    ticks = list(np.linspace(1, len(labels), len(labels)).astype(int))

    plt.xticks(ticks, labels)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    # Set max y. Min y is 1.0 and max the max value

    add_range = (max_val - min_val) * 0.10
    plt.ylim([min_val - add_range, max_val + add_range])

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
             ylabel='DetectionBoxes_Precision/mAP@.50IOU'):
    # mAP Visualization
    fig1, ax1 = plt.subplots(figsize=(7, 12))
    ax1.set_title(title)
    ax1.grid()

    ticks = list(np.linspace(0, len(labels) - 1, len(labels)).astype(int))

    # max_val = np.array(values).max() #np.max(max(values, key=tuple))
    # min_val = np.array(values).min() #np.min(min(values, key=tuple))
    # add_range = (max_val - min_val) * 0.10
    # plt.ylim([min_val - add_range, max_val + add_range])

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


def visualize_performance(df, output_dir, metric_precision='DetectionBoxes_Precision/mAP@.50IOU',
                          metric_recall='DetectionBoxes_Recall/AR@100'):
    '''


    '''

    values = list()
    labels = list()

    df_perf = df.sort_values(by=[metric_precision], ascending=False)

    for index, row in df_perf.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        values.append(row[metric_precision])
        # Add labels
        labels.append(network + " " + device)

    # max_val = np.max(values)
    # min_val = np.min(values)
    # plot_boxplot(values, labels, output_dir, title="Performance All Hardware", max_val=max_val, min_val=min_val)
    # plot_violin_plot(values, labels, output_dir, title="Performance All Hardware", max_val=max_val, min_val=min_val)

    # Extract important values
    data = {'Model': labels, metric_precision: values}
    df_table = pd.DataFrame(data)
    df_table.set_index('Model', inplace=True)
    df_table.round(2).astype(float).to_csv(
        os.path.join(output_dir, "Mean Average Precision".replace(' ', '_') + '_violinplot' + '.csv'),
        sep=";")

    # Visulization
    plot_bar(values, labels, output_dir, title="Mean Average Precision", xlabel="Models and Hardware",
             ylabel=metric_precision)

    # Recall Visualization
    df_perf = df.sort_values(by=[metric_recall], ascending=False)

    values = list()
    labels = list()

    for index, row in df_perf.iterrows():
        network = row['Model_Short']
        device = row['Hardware']
        print("Processing {} on {}".format(network, device))
        values.append(row[metric_recall])
        # Add labels
        labels.append(network + " " + device)

    plot_bar(values, labels, output_dir, title="Recall", xlabel="Models and Hardware", ylabel=metric_recall)


def visualize_performance_recall_optimum(latency, performance, output_dir, latency_requirement=None,
                                         performance_requirement=None, plot_baseline=False):
    '''
    Find the pareto optimum for models for mAP and latency

    '''

    if 'Date' in latency.columns:
        latency_reduced = latency.drop(columns=['Date'])
    else:
        latency_reduced = latency
    latency_reduced.set_index(['Model_Short', 'Hardware'])
    latency_reduced = latency_reduced[~latency_reduced.index.duplicated(keep='first')]
    if 'Date' in performance.columns:
        performance_reduced = performance.drop(columns=['Date'])
    else:
        performance_reduced = performance
    performance_reduced.set_index(['Model_Short', 'Hardware'])
    performance_reduced = performance_reduced[~performance_reduced.index.duplicated(keep='first')]

    lat_perf_df = pd.merge(latency_reduced, performance_reduced,
                           how='inner', left_index=True, right_index=True,
                           suffixes=("", "_extra"))
    lat_perf_df = lat_perf_df.reset_index()

    print("Available columns: ", lat_perf_df.columns)
    # Plot for all hardware
    print("Plot mAP vs Latency All Hardware")
    plot_performance_latency(lat_perf_df, output_dir, title='mAP to Latency All Hardware Full Range',
                             y_col='DetectionBoxes_Precision/mAP@.50IOU',
                             ylim=[0, 1],
                             latency_requirement=latency_requirement,
                             plot_baseline=plot_baseline)  # AP at IoU=.50:.05:.95 (primary challenge metric)
    print("Plot mAP vs Latency Zoom")
    plot_performance_latency(lat_perf_df, output_dir, title='mAP to Latency All Hardware',
                             y_col='DetectionBoxes_Precision/mAP@.50IOU',
                             latency_requirement=latency_requirement,
                             plot_baseline=plot_baseline)

    print("Plot Recall vs Latency")
    plot_performance_latency(lat_perf_df, output_dir, title='Recall to Latency All Hardware',
                             y_col='DetectionBoxes_Recall/AR@100',
                             latency_requirement=latency_requirement,
                             plot_baseline=plot_baseline)  # 100 Detections/image

    # Plot for each hardware separately
    unique_hardware = lat_perf_df['Hardware'].unique()
    for hw in unique_hardware:
        sub_df = lat_perf_df[lat_perf_df['Hardware'] == hw]
        plot_performance_latency(sub_df, output_dir, title='mAP to Latency ' + hw,
                                 y_col='DetectionBoxes_Precision/mAP@.50IOU',
                                 plot_separation='Network',
                                 latency_requirement=latency_requirement,
                                 plot_baseline=plot_baseline)
        plot_performance_latency(sub_df, output_dir, title='Recall to Latency ' + hw,
                                 y_col='DetectionBoxes_Recall/AR@100',
                                 plot_separation='Network',
                                 latency_requirement=latency_requirement,
                                 plot_baseline=plot_baseline)


def plot_performance_latency(lat_perf_df, output_dir=None, title='mAP_vs_Latency',
                             y_col='DetectionBoxes_Precision/mAP@.50IOU',
                             ylim=None, xlim=None, latency_requirement=None, plot_separation='Hardware',
                             plot_baseline=False):
    # Set font size
    plt.rcParams.update({'font.size': 14})

    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Get unique hardware
    hardware_types = list(lat_perf_df[plot_separation].unique())

    performance_max = np.max(lat_perf_df[y_col].values)
    performance_min = np.min(lat_perf_df[y_col].values)
    latency_max = np.max(lat_perf_df['Mean_Latency'].values)
    latency_min = np.min(lat_perf_df['Mean_Latency'].values)

    if lat_perf_df.shape[0] > 15:
        fig, ax = plt.subplots(figsize=[12, 12])
        print("More than 15 objects. Use large figure.")
    else:
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

    if latency_requirement:
        plt.vlines(int(latency_requirement), 0, 1.0, color='red')
        texts.append(
            plt.text(int(latency_requirement) + 2, 0.5, "Requirement {:.2f}ms".format(int(latency_requirement)),
                     rotation=90, verticalalignment='center'))

    if plot_baseline:
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


def get_performance_deltas_for_hardware_optimizations(performance, hwopt_reference=None):
    '''


    '''

    # For each model and hardware
    enhanced_performance = performance.copy()
    enhanced_performance['Relative_mAP'] = -1

    enhanced_performance['Custom_Parameters'] = enhanced_performance['Custom_Parameters'].astype(str) \
        .replace('None', '[]') \
        .replace('NaN', '[]') \
        .replace('nan', '[]')
    enhanced_performance['Hardware_Optimization'] = enhanced_performance['Hardware_Optimization'].astype(str) \
        .replace('None', '') \
        .replace('NaN', '') \
        .replace('nan', '')

    x = enhanced_performance.groupby(['Framework', 'Network', 'Resolution', 'Dataset', 'Custom_Parameters', 'Hardware'])

    if len(x) == len(enhanced_performance):
        warnings.warn("Something might be wrong with the grouping of results. "
                      "Hardware is grouped by Framework, Network, Resolution, Dataset, Custom_Parameters, Hardware."
                      "If the number of groups = number of results, then the HW optimizer matching has failed.")

    for name, group in x:
        print(name)
        print(group)

        if len(group[(pd.isnull(group['Hardware_Optimization'])) |
                     (group['Hardware_Optimization'] == hwopt_reference)]) > 0:
            original_performance = \
                group[(pd.isnull(group['Hardware_Optimization'])) |
                      (group['Hardware_Optimization'] == hwopt_reference)][
                    'DetectionBoxes_Precision/mAP@.50IOU'].values[0]

            for i, row in group.iterrows():
                current_performance = row['DetectionBoxes_Precision/mAP@.50IOU']
                relative_performance = np.round(current_performance / original_performance, 3)

                enhanced_performance.loc[(enhanced_performance['Framework'] == row['Framework']) &
                                         (enhanced_performance['Network'] == row['Network']) &
                                         (enhanced_performance['Resolution'] == row['Resolution']) &
                                         (enhanced_performance['Dataset'] == row['Dataset']) &
                                         (enhanced_performance['Custom_Parameters'] == row['Custom_Parameters']) &
                                         (enhanced_performance['Hardware'] == row['Hardware']) &
                                         (enhanced_performance['Hardware_Optimization'] == row[
                                             'Hardware_Optimization']),
                                         'Relative_mAP'] = relative_performance
                print("Created value for {}. Relative performance: {}".format(name, relative_performance))
        else:
            warnings.warn("No original performance for " + str(name))

    if any(i == -1 for i in enhanced_performance['Relative_mAP'].values):
        warnings.warn("There is no HW match for all values. -1 are still in the cells.")

    return enhanced_performance


def get_latency_deltas_for_hardware_optimizations(latency, hwopt_reference=None):
    '''
    Calculate relative latencies

    :argument



    '''

    # For each model and hardware
    enhanced_latency = latency.copy()
    enhanced_latency['Relative_Latency'] = -1

    enhanced_latency['Custom_Parameters'] = enhanced_latency['Custom_Parameters'].astype(str) \
        .replace('None', '[]') \
        .replace('NaN', '[]') \
        .replace('nan', '[]')
    enhanced_latency['Hardware_Optimization'] = enhanced_latency['Hardware_Optimization'].astype(str) \
        .replace('None', '') \
        .replace('NaN', '') \
        .replace('nan', '')

    # Group by network execution to compare all HW optimization methods should be compared to the no HW optimization method
    x = enhanced_latency.groupby(['Framework', 'Network', 'Resolution', 'Dataset', 'Custom_Parameters', 'Hardware'])
    if len(x) == len(enhanced_latency):
        warnings.warn("Something might be wrong with the grouping of results. "
                      "Hardware is grouped by Framework, Network, Resolution, Dataset, Custom_Parameters, Hardware."
                      "If the number of groups = number of results, then the HW optimizer matching has failed.")

    for name, group in x:
        print(name)
        print(group)

        # Look for executions without any hardware optimizations to use to compare to
        if len(group[(pd.isnull(group['Hardware_Optimization'])) | (
                group['Hardware_Optimization'] == hwopt_reference)]) > 0:
            original_latency = \
                group[
                    (pd.isnull(group['Hardware_Optimization'])) | (group['Hardware_Optimization'] == hwopt_reference)][
                    'Mean_Latency'].values[0]

            for i, row in group.iterrows():
                current_latency = row['Mean_Latency']
                relative_latency = np.round(current_latency / original_latency, 3)

                enhanced_latency.loc[(enhanced_latency['Framework'] == row['Framework']) &
                                     (enhanced_latency['Network'] == row['Network']) &
                                     (enhanced_latency['Resolution'] == row['Resolution']) &
                                     (enhanced_latency['Dataset'] == row['Dataset']) &
                                     (enhanced_latency['Custom_Parameters'] == row['Custom_Parameters']) &
                                     (enhanced_latency['Hardware'] == row['Hardware']) &
                                     (enhanced_latency['Hardware_Optimization'] == row['Hardware_Optimization']),
                                     'Relative_Latency'] = relative_latency
                print("Created value for {}. Relative latency: {}".format(name, relative_latency))
        else:
            warnings.warn("No original latency for {}" + str(name))

    if any(i == -1 for i in enhanced_latency['Relative_Latency'].values):
        warnings.warn("There is no HW match for all values. -1 are still in the cells.")

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

        if len(values) > 0 and len(values[1:]) > 0:
            max_val = np.max(
                np.array([np.max(x) for x in values]))  # np.array(values[1:]).max()  # np.max(max(values, key=tuple))
            min_val = np.min(
                np.array([np.min(x) for x in values]))  # np.array(values[1:]).min()  # np.min(min(values, key=tuple))

            plot_boxplot(values[1:], labels[1:], output_dir, title=title + " for " + hw_name, xlabel=xlabel,
                         ylabel=ylabel, max_val=max_val, min_val=min_val)
            plot_violin_plot(values[1:], labels[1:], output_dir, title=title + " for " + hw_name, xlabel=xlabel,
                             ylabel=ylabel, max_val=max_val, min_val=min_val)
        # else:
        #    first_values = []
        #    for e in values:
        #        first_values.append(e[0])
        #    plot_bar(first_values, labels, output_dir, title=title + " for " + hw_name, xlabel=xlabel,
        #             ylabel=ylabel)


def evaluate(latency_file, performance_file, output_dir, hwopt_reference=None, input_combined_file=None,
             latency_requirement=None, performance_requirement=None, plot_baseline=False):
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

    # latency['Latencies'] = latency['Latencies'].astype(object).replace(np.nan, None)
    latency = latency.replace({np.nan: None})

    if latency['Mean_Latency'][0] < 1:
        warnings.warn("Mean latency <0. No network is so fast. Probably the unit is s. Converting to ms.")
        latency['Mean_Latency'] = latency['Mean_Latency'] * 1000
        if not latency['Latencies'][0] is None:
            for index, row in latency.iterrows():
                latency['Latencies'][index] = str(list(np.array(ast.literal_eval(row['Latencies'])) * 1000))

    relative_latencies = get_latency_deltas_for_hardware_optimizations(latency, hwopt_reference)
    visualize_relative_latencies(relative_latencies, output_dir)

    # Visualize latency
    visualize_latency(latency, output_dir)

    if performance is not None:
        # Visualize relative performance
        relative_performance = get_performance_deltas_for_hardware_optimizations(performance, hwopt_reference)
        visualize_relative_latencies(relative_performance, output_dir, measurement='Relative_mAP',
                                     title="Relative Performance",
                                     ylabel="Relative Performance [mAP]/([mAP] Original)"
                                     )
        # Visualize Performance
        visualize_performance(performance, output_dir)

        # mAP/latency curve
        visualize_performance_recall_optimum(latency, performance, output_dir, latency_requirement, plot_baseline)


if __name__ == "__main__":
    if args.performance_file or args.latency_file:
        raise SystemExit("Use combined_results file instead of separate latency and performance files.")

    evaluate(None, None, args.output_dir, args.hwopt_reference, args.input_combined_file,
             args.latency_requirement, args.performance_requirement, args.plot_baseline)
    # visualize_values(args.infile)

    print("=== Program end ===")
