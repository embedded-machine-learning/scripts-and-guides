echo #====================================#
echo # clean dataset
echo #====================================#

echo Clean images from images with wrong formats, which causes error in the training phase

python clean_images.py ^
-i "images"