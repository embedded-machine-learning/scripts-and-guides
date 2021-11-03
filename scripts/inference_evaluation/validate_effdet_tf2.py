import inference
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

imgs, elapsed_list = [], []
image_path_pattern = 'C:/Projekte/21_SoC_EML/datasets/oxford-pets/images/val_debug/*.jpg'
output_dir = 'testdata/output'
image_size = [1024, 1024]
batch_size = 1
min_score_thresh = 0.1
max_boxes_to_draw = 60

# read input images
for f in tf.io.gfile.glob(image_path_pattern):
  imgs.append(np.array(Image.open(f)))

# set up driver with given parameters
driver = inference.ServingDriver(
  'efficientdet-d0', 'efficientdet-d0', batch_size=batch_size)
driver.build(params_override=dict(image_size=image_size), min_score_thresh=min_score_thresh, max_boxes_to_draw=max_boxes_to_draw)

# run inference on each image, visualize and save output
for i, input in enumerate(imgs):
  start_time = time.time()
  predictions = driver.serve_images([input])
  img = driver.visualize(input, predictions[0],
                         line_thickness=1)
  output_image_path = os.path.join(output_dir, str(i) + '.jpg')
  Image.fromarray(img).save(output_image_path)
  elapsed_time = time.time() - start_time
  elapsed_list.append(elapsed_time)
  print("--- %s seconds ---" % elapsed_time)

print("Mean elapsed time:", sum(elapsed_list)/len(elapsed_list))