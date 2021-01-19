echo #######################################
echo # Rename images                       #
echo #######################################

python 012_rename_images.py ^
--image_dir="samples/images" ^
--new_name="test_frame_"

dir samples\images

python 012_rename_images.py ^
--image_dir="samples/images" ^
--new_name="frame_"
--start_number=0

dir samples\images
