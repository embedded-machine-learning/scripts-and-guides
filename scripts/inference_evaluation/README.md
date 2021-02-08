<div align="center">
  <img src="../../_img/eml_logo_and_text.png">
</div>

# Evaluation Tools

The evaluation tools contain useful tools for evaluating the performance of inference. As the scripts were usually not written from scratch, but copied 
and modified from other sources, the licence information of the other sources is kept within the script.

## Tools

### Visualize Bounding Boxes with TensorFlow 2
File: obj_visualize_compare_bbox.py
Usage: Select two or three images with their bounding boxes in PASCAL VOC XML format and visualize then within one image.
Example as Windows batch script: 
```
python obj_visualize_compare_bbox.py --labelmap="samples/annotations/label_map.pbtxt" ^
--output_dir="samples/results" ^
--image_path1="samples/images/0.jpg" --annotation_dir1="samples/annotations/xml" --title1="Image 1" ^
--image_path2="samples/images/10.jpg" --annotation_dir2="samples/annotations/xml" --title2="Image 2" ^
--image_path3="samples/images/20.jpg" --annotation_dir3="samples/annotations/xml" --title3="Image 3" ^
--use_three_images
```

Result for comparing three different images.

<div align="center">
  <img src="./samples/results/bbox_0_10_20.jpg">
</div>

### Visualize Bounding Boxes with OpenCV
File: visualize_object_detection_images_opencv.py
Usage: Select two images with their bounding boxes in PASCAL VOC XML format and visualize then within one image.
It uses OpenCV for the visualization of the bounding boxes.
Example as Windows batch script: 
```
python visualize_object_detection_images_opencv.py ^
--image_path1="samples/images/0.jpg" --annotation_dir1="samples/annotations/xml" ^
--image_path2="samples/images/30.jpg" --annotation_dir2="samples/annotations/xml" ^
--output_dir="samples/results" ^
--line_thickness=2
```

### Perform Inference TensorFlow 2 Saved Model
Usage: Select two images with their bounding boxes in PASCAL VOC XML format and visualize then within one image.
It uses OpenCV for the visualization of the bounding boxes.

Script: tf2oda_inference_from_saved_model.py

Source: 

Example as Windows batch script: 
```
:: Constants Definition
set MODELNAME=ssd_mobilenet_v2_R300x300_D100_coco17_starwars
set SCRIPTPREFIX=.


python %SCRIPTPREFIX%\tf2oda_inference_from_saved_model.py ^
--model_path "../training/samples/starwars_reduced/exported-models/%MODELNAME%/saved_model/" ^
--image_dir "../training/samples/starwars_reduced/images/test" ^
--labelmap "../training/samples/starwars_reduced/annotations/sw_label_map.pbtxt" ^
--output_dir="../training/samples/starwars_reduced/result/%MODELNAME%" ^
--run_detection True 
```


## Issues
Should any issues arise during the completion of the guide or any errors noted, please let us know by filing an issue and help us keep up the quality.
