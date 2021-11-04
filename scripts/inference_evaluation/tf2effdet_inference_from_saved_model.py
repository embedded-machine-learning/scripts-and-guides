#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Infer Google AutoML TF2 EfficientDet, latency and detection boxes.

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
# Source:

"""

# Futures
# from __future__ import print_function

# Built-in/Generic Imports
import argparse
import logging
import json
import warnings

# Libs
import numpy as np
import os
import time
import pandas as pd

# Own modules
import image_utils as im
import inference_utils as inf
# Include AutoML EfficientDet Inference, not TF2ODA
import inference

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(description='Google AutoML EfficientDet Inferrer')
parser.add_argument("-p", '--model_path',
                    default='C:/Projekte/21_SoC_EML/eml_projects/efficientdet-oxford-pets/exported-models/tf2_efficientdet-d1_512x512_oxfordpets/saved_model',
                    help='Saved model path', required=False)
parser.add_argument("-i", '--image_dir', default='C:/Projekte/21_SoC_EML/datasets/oxford-pets/images/val_debug/',
                    help='Saved model path', required=False)
parser.add_argument("-s", '--min_score', default=0.5, type=float,
                    help='Min score of detection box to save the image.', required=False)
parser.add_argument('-b', '--batch_size', type=int, default=1,
                    help='Batch Size', required=False)
parser.add_argument('-is', '--image_size', type=str, default=None,
                    help='List of two coordinates: [Height, Width]', required=False)
parser.add_argument("-ms", '--model_short_name', default=None, type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-m", '--model_name', default="tf2_efficientdet-d1_512x512_oxfordpets", type=str,
                    help='Model name for collecting model data.', required=False)
parser.add_argument("-hw", '--hardware_name', default="Inteli7", type=str,
                    help='Hardware name collecting statistical data.', required=False)

parser.add_argument("-l", '--labelmap', default='annotations/mscoco_label_map.pbtxt.txt',
                    help='Labelmap path', required=False)
parser.add_argument("-lr", '--latency_runs', default=1000, type=int,
                    help='Number of runs for latency check', required=False)

parser.add_argument("-out", '--detections_out', default='./results/detections.csv',
                    help='Output file detections', required=False)
parser.add_argument("-lat", '--latency_out', default="./results/latency.csv", help='Output path for latencies file, which is '
                                                                         'appended or created new. ', required=False)
parser.add_argument('-id', '--index_save_file', type=str, default='./tmp/index.txt',
                    help='Path to put index file to keep the same key for different types of measurements.',
                    required=False)

#parser.add_argument('-si', '--detected_images_dir', type=str, default=None,
#                    help='If this parameter is not None, but a folder, detection boxes are plotted into the image'
#                         'with the class',
#                    required=False)
#parser.add_argument('-mop', '--model_optimizer_prefix', type=str, default='TRT',
#                    help='Prefix for Model Optimizer Settings', required=False)

args = parser.parse_args()

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

log.info(args)

def batch_input(batch_size, data_path, d_type, hw, is_keras=False):
    '''
    Create one representative batch out of the dataset


    TODO: Use all images in the dataset to create a batch, not only the first image, i.e. create
    TODO: a batch as in reality

    '''

    if d_type == 'float32':
        datatype = np.float32
    elif d_type == 'float16':
        datatype = np.float16
    elif d_type == 'uint8':
        datatype = np.uint8
    else:
        raise ValueError("No valid data type provided: " + d_type + ". It has to be float32, float16 or uint8")

    batched_input = np.zeros((batch_size, hw[0], hw[1], 3), dtype=datatype)

    #batched_input2 = np.array(Image.open("C:/Projekte/21_SoC_EML/eml_projects/efficientdet-oxford-pets/results/test/0.jpg"))

    # if os.path.isfile(data_path):
    #     pics = data_path
    # else:
    #     pics = os.listdir(data_path)
    # n = len(pics)
    #
    # for i in range(batch_size):
    #     if os.path.isfile(data_path):
    #         img_path = data_path
    #     else:
    #         img_path = os.path.join(data_path, pics[i % n])  # generating batches
    #     img = image.load_img(img_path, target_size=(hw[0], hw[1]))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     if is_keras:
    #         x = preprocess_input(x)  # for models loaded from Keras applications
    #     batched_input[i, :] = x
    #
    # batched_input = tf.constant(batched_input)
    return batched_input


def infer_latency(driver, image_dir, hardware_name, model_name, model_short_name, latency_out,
                  N_warmup_run=50, N_run=1000, batch_size=1, d_type='uint8', image_size=[300, 300],
                  index_save_file="./tmp/index.txt"):
    '''



    '''

    input = batch_input(batch_size, image_dir, d_type, image_size, is_keras=False)

    elapsed_time = []
    # all_preds = []
    # boxes = []
    # classes = []
    # scores = []
    # batch_size = batched_input.shape[0]

    print("Running warm up runs...i.e. just running empty runs to load the model correctly")
    for i in range(N_warmup_run):
        labeling = driver.serve_images(input) #infer(input)
        # print("Inference {}/{}".format(i, N_warmup_run))
        # preds = labeling['predictions'].numpy()
        preds = labeling

    print("Running real runs with one batch to create the images...")
    for i in range(N_run):
        start_time = time.time()
        labeling = driver.serve_images(input)
        # preds = labeling['predictions'].numpy()
        preds = labeling
        end_time = time.time()

        latency = (end_time - start_time) * 1000    #in ms

        elapsed_time.append(latency)
        # elapsed_time = np.append(elapsed_time, end_time - start_time)

        # all_preds.append(preds)

        if i % 50 == 0:
            print('Steps {}-{} average: {:4.1f}ms'.format(i, i + 50, (np.array(elapsed_time[-50:]).mean())))

    # throughput = N_run * batch_size / elapsed_time.sum()

    # Create the latency.csv file
    # Generate identifier for this run
    index = inf.generate_measurement_index(model_name)
    inf.save_latencies_to_csv(elapsed_time, batch_size, N_run, hardware_name, model_name, model_short_name, latency_out,
                          index)
    #Save index to a file
    file1 = open(index_save_file, 'w')
    file1.write(index)
    print("Index {} used for latency measurement".format(index))

def load_model(saved_model_dir, model_name, batch_size=1):
    """
    Load saved model

    """

    # set up driver with given parameters
    driver = inference.ServingDriver(
        model_name,
        None,
        batch_size=batch_size)

    # driver.build(params_override=dict(image_size=image_size),
    #             min_score_thresh=min_score_thresh,
    #             max_boxes_to_draw=max_boxes_to_draw)

    driver.load(saved_model_dir)
    print("Loaded model from: ", saved_model_dir)

    return driver


def detect_image(driver, image_path):
    """
    Detect inferred image

    """

    print("Process ", image_path)
    image_filename = os.path.basename(image_path)

    total_latency_start_time = time.time()
    # Load image
    image_np = im.load_image_into_numpy_array(image_path)

    # Make image tensor of it
    input_tensor = np.expand_dims(image_np, 0)


    start_time = time.time()  # Start time measurement
    # Infer
    #predictions = driver.serve_images([input])
    predictions = driver.serve_images(input_tensor)
    latency = time.time() - start_time  # Stop time measurement
    print("Inference time {} : {}s".format(image_filename, latency))

    prediction = predictions[0]

    # Process detections
    # width image_np.shape[1], height image_np.shape[0]
    boxes = prediction[:, 1:5]
    # Values are returned as pixels. Convert to quotes.
    boxes_norm = boxes/np.array([image_np.shape[0], image_np.shape[1], image_np.shape[0], image_np.shape[1]])
    classes = prediction[:, 6].astype(int)
    scores = prediction[:, 5]

    # Process detections
    #boxes = detections['detection_boxes'][0].numpy()
    #classes = detections['detection_classes'][0].numpy().astype(np.int32)
    #scores = detections['detection_scores'][0].numpy()

    total_latency_stop_time = time.time()
    total_latency = total_latency_stop_time - total_latency_start_time
    print("Total latency for {} : {}s".format(image_filename, total_latency))

    return image_filename, image_np, prediction, boxes_norm, classes, scores, latency, total_latency


def infer_images(model_path, image_dir, latency_out, detections_out, min_score, model_name,
                 hardware_name, model_short_name=None, batch_size=1, image_size=None,
                 model_optimizer_prefix='TRT', index_save_file="./tmp/index.txt", latency_runs=1000):
    """



    """

    # Create output directories
    os.makedirs(os.path.dirname(detections_out), exist_ok=True)
    os.makedirs(os.path.dirname(latency_out), exist_ok=True)
    os.makedirs(os.path.dirname(index_save_file), exist_ok=True)

    # Get model infos
    model_info = inf.get_info_from_modelname(model_name, model_short_name,
                                             model_optimizer_prefix=model_optimizer_prefix)
    print("Model information: ", model_info)
    if image_size:
        image_size = json.loads(image_size)
        if (image_size[0] != model_info['resolution'][0]) or (image_size[1] != model_info['resolution'][1]):
            warnings.warn("Provided input resolution differs from model resolution: "
                          "Input={}, model={}".format(image_size, model_info['resolution']))
        else:
            print("Using image resolution {}".format(image_size))

    else:
        image_size = model_info['resolution']
        print("In the batch processing, model resolution {} will be used".format(image_size))

    imgs, elapsed_list = [], []
    # image_path_pattern = 'C:/Projekte/21_SoC_EML/datasets/oxford-pets/images/val_debug/*.jpg'
    # saved_model_dir = 'C:/Projekte/21_SoC_EML/eml_projects/efficientdet-oxford-pets/exported-models/tf2_efficientdet-d1_512x512_oxfordpets/saved_model'
    # model_name = 'efficientdet-d0'

    # Allowed models from hparams_config.py in the AutoML repo
    allowed_models = ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4',
                      'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7', 'efficientdet-d7x', 'efficientdet-lite0',
                      'efficientdet-lite1', 'efficientdet-lite2', 'efficientdet-lite3', 'efficientdet-lite3x',
                      'efficientdet-lite4']
    model_type = model_info['network']
    if not any(x == model_type for x in allowed_models):
        raise Exception("Model name {} does not exist. It has to be one of {}".format(model_type, allowed_models))

    output_dir = 'results/test'
    # image_size = [1024, 1024]
    # batch_size = 1
    # min_score_thresh = 0.1
    # max_boxes_to_draw = 60

    # Load model
    driver = load_model(model_path, model_type, batch_size=batch_size)

    print("Perform latency tests.")
    infer_latency(driver, image_dir, hardware_name, model_name, model_info['model_short_name'], latency_out,
                  N_warmup_run=50, N_run=latency_runs, batch_size=batch_size, d_type='uint8',
                  image_size=image_size, index_save_file=index_save_file)


    # Load inference images
    print("Loading images from ", image_dir)
    image_list = im.get_images_name(image_dir)

    # read input images and append as array
    #for f in tf.io.gfile.glob(os.join(image_dir, '*.jpg')):
    #    imgs.append(np.array(Image.open(f)))

    # Define scores and latencies
    latencies = []
    total_latencies = []
    detection_scores = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin',
                                             'ymin', 'xmax', 'ymax', 'score'])
    # Process each image
    for image_name in image_list:
        # if run_detection:
        image_filename, image_np, prediction, boxes, classes, scores, latency, total_latency = \
            detect_image(driver, os.path.join(image_dir, image_name))
        latencies.append(latency)
        total_latencies.append(total_latency)

        bbox_df = inf.convert_reduced_detections_tf2_to_df(image_filename, image_np, boxes, classes, scores, min_score)
        detection_scores = detection_scores.append(bbox_df)

        # Plot image if detected_images_dir is set to a folder
        #if detected_images_dir and os.path.isdir(detected_images_dir):
        #    img = driver.visualize(image_np[0], prediction, line_thickness=1)
        #    new_name = os.path.basename(image_name) + '_detections.jpg'
        #    output_image_path = os.path.join(detected_images_dir, new_name)
        #    Image.fromarray(img).save(output_image_path)
        #    print("Saved image: ", output_image_path)


    print("Mean latency without batch processing: {}".format(np.array(latencies[1:-1]).mean()))
    print("Mean total latency including image preprocessing: {}".format(np.array(total_latencies[1:-1]).mean()))

    # Save all detections
    # if run_detection and xml_dir and detection_scores.shape[0] > 0:
    # Save detections
    detection_scores.to_csv(detections_out, index=None, sep=';')
    print("Detections saved to ", detections_out)



    # # run inference on each image, visualize and save output
    # for i, input in enumerate(imgs):
    #     # print("Process image: ", input)
    #     start_time = time.time()  # Start time measurement
    #     # Infer
    #     predictions = driver.serve_images([input])
    #     elapsed_time = time.time() - start_time  # Stop time measurement
    #
    #     prediction = predictions[0]
    #
    #     boxes = prediction[:, 1:5]
    #     classes = prediction[:, 6].astype(int)
    #     scores = prediction[:, 5]
    #
    #     img = driver.visualize(input, predictions[0], line_thickness=1)
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_image_path = os.path.join(output_dir, str(i) + '.jpg')
    #     Image.fromarray(img).save(output_image_path)
    #
    #     elapsed_list.append(elapsed_time)
    #     print("--- %s seconds ---" % elapsed_time)
    #
    # print("Mean elapsed time:", sum(elapsed_list) / len(elapsed_list))


# def saved_model_inference(self, image_path_pattern, output_dir, **kwargs):
#     """Perform inference for the given saved model."""
#     driver = inference.ServingDriver(
#         self.model_name,
#         self.ckpt_path,
#         batch_size=self.batch_size,
#         use_xla=self.use_xla,
#         model_params=self.model_config.as_dict(),
#         **kwargs)
#     driver.load(self.saved_model_dir)
#
#     # Serving time batch size should be fixed.
#     batch_size = self.batch_size or 1
#     all_files = list(tf.io.gfile.glob(image_path_pattern))
#     print('all_files=', all_files)
#     num_batches = (len(all_files) + batch_size - 1) // batch_size
#
#     for i in range(num_batches):
#         batch_files = all_files[i * batch_size:(i + 1) * batch_size]
#         height, width = self.model_config.image_size
#         images = [Image.open(f) for f in batch_files]
#         if len(set([m.size for m in images])) > 1:
#             # Resize only if images in the same batch have different sizes.
#             images = [m.resize(height, width) for m in images]
#         raw_images = [np.array(m) for m in images]
#         size_before_pad = len(raw_images)
#         if size_before_pad < batch_size:
#             padding_size = batch_size - size_before_pad
#             raw_images += [np.zeros_like(raw_images[0])] * padding_size
#
#         detections_bs = driver.serve_images(raw_images)
#         for j in range(size_before_pad):
#             img = driver.visualize(raw_images[j], detections_bs[j], **kwargs)
#             img_id = str(i * batch_size + j)
#             output_image_path = os.path.join(output_dir, img_id + '.jpg')
#             Image.fromarray(img).save(output_image_path)
#             print('writing file to %s' % output_image_path)


if __name__ == "__main__":
    infer_images(args.model_path, args.image_dir, args.latency_out, args.detections_out, args.min_score,
                 args.model_name, args.hardware_name, model_short_name=args.model_short_name,
                 batch_size=args.batch_size, image_size=args.image_size,
                 model_optimizer_prefix=None, index_save_file=args.index_save_file, latency_runs=args.latency_runs)

    print("=== Program end ===")
