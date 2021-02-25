echo #######################################
echo # Rename images                       #
echo #######################################

python rename_images.py ^
--image_dir="samples/images" ^
--new_name="test_frame_"

dir samples\images

python rename_images.py ^
--image_dir="samples/images" ^
--new_name="frame_"
--start_number=0

dir samples\images
