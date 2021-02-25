echo #######################################
echo # Convert CVML to Coco                #
echo #######################################

echo Warning: Do not forget to modify the script for the image file name structure

python C:\Projekte\21_SoC_EML\public_content\scripts-and-guides\scripts\conversion\convert_3DMOT2015_yololike_to_coco.py ^
--annotation_file="samples/annotations/3DMOT2015_yololike_ground_truth.txt" ^
--image_dir="samples/yolo_images" ^
--label_name="pedestrian" ^
--image_name_prefix=""
