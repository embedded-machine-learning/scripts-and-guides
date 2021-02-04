import argparse

import os
import sys
import numpy as np
import time

from multiprocessing import Pool

import matplotlib
#If you get _tkinter.TclError: no display name and no $DISPLAY environment variable use
# matplotlib.use('Agg') instead
matplotlib.use('TkAgg')


from six import BytesIO

import re
import pickle

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def load_labelmap(path):
    '''

    :param path:
    :return:
    '''

    #labelmap = label_map_util.load_labelmap(path)
    category_index = label_map_util.create_category_index_from_labelmap(path)

    return category_index

#def make_windows_path(path):
#    '''#


#    :param path:
#    :return:
#    '''
    #Select paths based on OS
    #if (os.name == 'nt'):
    #    print("Windows system")
        #Windows paths
        #mo_file = os.path.join("C://", "Program Files (x86)", "IntelSWTools", "openvino", "deployment_tools",
        #                       "model_optimizer", "mo.py")

    #    new_path = os.path.join(path)

        #path = path.strip() #replace("//", "/")
    #else:
    #    new_path = path

#    if not (os.path.isdir(path) or os.path.isfile(path)):
#        print("File or folder does not exist. Add current path")
#        new_path = os.path.join(os.getcwd(), new_path)

#    return new_path

def load_model(model_path):
    '''


    :param model_path:
    :return:
    '''
    start_time = time.time()
    print("Start model loading from path ", model_path)
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(model_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Finished. Elapsed time: ' + str(elapsed_time) + 's')

    return detect_fn


def create_image_namelist(source):
    source = source.replace('\\', '/')
    image_names = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]
    return image_names

def create_single_imagedict(source,image_name):
    image_dict = {}
    image_path = os.path.join(source, image_name)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    image_dict[image_name] = (image_np, input_tensor)
    return image_dict


def load_images(source):
    '''


    :param source:
    :return:
    '''
    #print(source)

    source = source.replace('\\', '/')
    image_names = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    #print(image_names) 
    #input()
    image_dict = dict()
    for image_name in image_names:
        image_path = os.path.join(source, image_name)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = np.expand_dims(image_np, 0)
        image_dict[image_name] = (image_np, input_tensor)

        #plt.imshow(image_np)
        #plt.show()

    return image_dict

def detect_images(detect_fn, image_dict):
    '''


    :param detect_fn:
    :param image_dict:

    :return:
    '''
    elapsed = []
    detection_dict = dict()

    #print("Start detection")
    for image_name, value in image_dict.items():
        image_np, input_tensor = value
        start_time = time.time()
        detections = detect_fn(input_tensor)
        end_time = time.time()
        diff = end_time - start_time
        elapsed.append(diff)
        print("Inference time image {} : {}s".format(image_name, diff))

        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),

        detection_dict[image_name] = (image_np, detections['detection_boxes'][0].numpy(),
                                      detections['detection_classes'][0].numpy().astype(np.int32),
                                      detections['detection_scores'][0].numpy())

    mean_elapsed = sum(elapsed) / float(len(elapsed))
    #print('Mean elapsed time: ' + str(mean_elapsed) + 's/image')

    return detection_dict

def visualize_images(detection_array, category_index, model_name, output_dir):
    '''

    :param detection_array:
    :param category_index:
    :return:
    '''

    image_dir = output_dir + "/" + model_name + "_inference"
    if os.path.isdir(image_dir)==False:
        os.makedirs(image_dir)
        print("Created directory {}".format(image_dir))

    #print("Visualize images")

    for image_name, value in detection_array.items():
        # Get objects
        image_np, boxes, classes, scores = value

        if(max(scores)>=0.85):
            print("Visualize image")
            print(image_name)
            #print(value)
            #print(classes)
            #print(scores)
            #print(boxes)
            #input()

            plt.rcParams['figure.figsize'] = [42, 21]
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.40,
                agnostic_mode=False)
            #plt.show()
            #plt.subplot(5, 1, 1)
            plt.imshow(image_np_with_detections)

            plt.savefig(image_dir + "/" + image_name + "_detections_" + model_name + ".png")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Google Tensorflow Detection API 2.0 Inferrer')
    parser.add_argument("-p", '--model_path', default='pre-trained-models/efficientdet_d5_coco17_tpu-32/saved_model/',
                        help='Saved model path', required=False)
    parser.add_argument("-i", '--image_dir', default='images/inference',
                        help='Saved model path', required=False)
    parser.add_argument("-l", '--labelmap', default='annotations/mscoco_label_map.pbtxt.txt',
                        help='Labelmap path', required=False)
    parser.add_argument("-r", '--run_detection', default=False,
                        help='Run detection or load saved detection model', required=False, type=bool)
    parser.add_argument("-o", '--output_dir', default="result",
                        help='Result directory', required=False)

    #C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/inference_efficientnet/

    args = parser.parse_args()

    model_name = args.model_path.split('/')[-3]

    #p = make_windows_path(args.model_path)

    #Load label path
    category_index = load_labelmap(os.path.abspath(args.labelmap))

    if args.run_detection:

        #Load model
        print("Loading model...")
        detector = load_model(args.model_path)

        # Load inference images
        print("Loading images...")
        #images_array = load_images(args.image_dir)

        imagelist = create_image_namelist(args.image_dir)

        #print(imagelist)
        
        for image_name in imagelist:
            pic_time = time.time()
            #print(image_name)
            image_dict = create_single_imagedict(args.image_dir,image_name)

            detection = detect_images(detector,image_dict)

            visualize_images(detection,category_index,model_name,args.output_dir)

            pic_end = time.time()
            duration = pic_end - pic_time
            print(image_name+ " took:"+str(duration)+"seconds")

            #print(image_dict)

        #print(images_array)

    end_time = time.time()

    elapsed = end_time -start_time
    print("This took:"+str(elapsed)+"seconds")

        



if __name__ == "__main__":
    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")
    main()

    print("=== Program end ===")
