# Data Processing Tools
The following data processing tools are used to rename files, check duplicates, partition datasets.

For each python script, there is a batch or sh script to demonstrate how it works.

## Pre-requisites
TBD

## Script Setup
In several scripts, there is a part for setting the constants and setting up the environment. 

```shell
echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=.\

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%
```

## Find Duplicate Images
This scripts looks for duplicate images within a folder and if such an image is found, it is moved to a new folder with duplicates.

Source: 

Script: `check_duplicate_images.py` 

Example: 
```shell
python check_duplicate_images.py ^
--train_imgs_dir "samples/images"
```

## Clean Images
This scripts looks at encodings for images or RGB color schemes and unifies them. It is used for tensorflow. In case of "strange" encodings, you get an error
in the training phase.

Source: 

Script: `clean_images.py` 

Example: 
```shell
python clean_images.py ^
-i "images"
```

## Partition Dataset
This scripts partitions a dataset in a training and a validation dataset. In the image folder, it creates a train and a val folder for the images.
The following parameters are used:
- -i: image dir
- --xmlDir: PASCAL VOC xml files
- --file_id_dir: annotations directory
- -r: Partition of images for validation set
- --remove source: Moves the images instead of copying

Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

Script: `partition_dataset.py` 

Example: 
```shell
python partition_dataset.py ^
-i "images" ^
--xmlDir "annotations/xmls" ^
-r 0.10 ^
--remove_source
```

## Reaname Image Filenames Within Pascal VOC XML Files
This script opens all xml files in a folder with PASCAL VOC data formats. It searches for the tag "filename" and renames the images there. It is used if e.g. 
the images are called "abc_0001.jpg" and in the xml files they are called "0001.jpg". The image prefix is set e.g. "abc_". Then the new xml file is stored in
the output_folder. 

Notice: The file name count has to be 4 digit (use zfill(4)).

Source: https://gist.github.com/jinyu121/a222492405890ce912e95d8fb5367977

Script: `rename_image_in_xml.py` 

Example: 
```shell
python rename_image_in_xml.py ^
--annotation_folder="samples/tmp" ^
--image_prefix="test_" ^
--output_folder="samples/tmp/renamed"
```

## Rename all images in a Folder
This script renames all images of a folder --image_dir to the format [prefix]_[count].[extension]. The --new_name parameter sets the file name prefix. 
Images automatically get a 4 digit count, e.g. 0001.jpg instead of 1.jpg. The script considers the order of the images if they are part of frames as long 
as the images have a 4 digit counting format. To pad the images to this counting format, plaese use a name padding script.

Notice: The file name count has to be 4 digit (use zfill(4)).

Source: 

Script: `rename_images.py` 

Example: 
```shell
python rename_images.py ^
--image_dir="samples/images" ^
--new_name="test_frame_"
```

## Rename Images by Padding Image Counts
This script renames all files of a folder --files_dir by padding the counters. Example: Images with the names abc_1.jpg are renames to abc_0001.jpg.

Source:

Script: `rename_pad_files.py` 

Example: 
```shell
python rename_pad_files.py ^
--files_dir="samples/tmp"
```

## Remove Invalid Bounding Boxes from Pascal VOC XML Files
This script checks if the bounding boxes of an image are inside of the image. If not, the bounding box is removed. Such bounding boxes have to be removed, esle
the training and validation does not work well.

Source:

Script: `remove_invalid_bounding_boxes_in_xml.py` 

Example: 
```shell
python %SCRIPTPREFIX%/remove_invalid_bounding_boxes_in_xml.py ^
--annotation_folder="annotations" ^
--output_folder="annotations/cleaned"
```

## Todos
1. Create a script that looks for images and annotations with multiple \".\" in the filename and replace them in the images as well as inside the Pascal VOC xml files.

# Issues
If there are any issues or suggestions for improvements, please add an issue to github's bug tracking system or please send a mail 
to [Alexander Wendt](mailto:alexander.wendt@tuwien.ac.at)

<div align="center">
  <img src="../../_img/eml_logo_and_text.png", width="500">
</div>
