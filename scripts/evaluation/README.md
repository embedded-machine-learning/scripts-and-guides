<div align="center">
  <img src="./images/eml_logo_and_text.png">
</div>

# Evaluation Tools

The evaluation tools contain useful tools for evaluating the performance of inference.

## Tools
File: obj_visualize_compare_bbox.py
Usage: Select two or three images with their bounding boxes in PASCAL VOC XML format and visualize then within one image.
Example: 
'''python obj_visualize_compare_bbox.py --labelmap="samples/annotations/label_map.pbtxt" ^
--output_dir="samples/results" ^
--image_path1="samples/images/0.jpg" --annotation_dir1="samples/annotations/xml" --title1="Image 1" ^
--image_path2="samples/images/10.jpg" --annotation_dir2="samples/annotations/xml" --title2="Image 2" ^
--image_path3="samples/images/20.jpg" --annotation_dir3="samples/annotations/xml" --title3="Image 3" ^
--use_three_images'''
