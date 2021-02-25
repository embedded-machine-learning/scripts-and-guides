echo #====================================#
echo # Validate and clean dataset
echo #====================================#

echo Check if there are duplicate images in a folder and move the duplicates to another folder

python check_duplicate_images.py --train_imgs_dir "samples/images"