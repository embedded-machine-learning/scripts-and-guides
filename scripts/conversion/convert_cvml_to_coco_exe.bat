echo #######################################
echo # Convert CVML to Coco                #
echo #######################################

echo Warning: Do not forget to modify the script for the image file name structure

python convert_cvml_to_coco.py ^
--annotation_file="samples/annotations/cvml_xml/cvml_Milan-PETS09-S2L1.xml" ^
--image_dir="samples/cvml_images" ^
--label_name="pedestrian" ^
--image_name_prefix="frame_"
