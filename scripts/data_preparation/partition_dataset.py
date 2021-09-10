""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
  -x, --xml_copy             Set this flag if you want the xml annotation files to be processed and copied over.
  -d, --xmlDir XMLDIR    Path where xml files are located
  -m, --remove_source   Remove the source images
  

  source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
"""

import os
import re
from shutil import copyfile
import argparse
import math
import random


def iterate_dir(source, dest, ratio, copy_xml, xml_directory, remove_source):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    xml_directory = xml_directory.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'val')
    inference_dir = os.path.join(dest, 'inference')

    if xml_directory is None:
        xml_source = source
        print("No XML folder specified. Use ", xml_source)
    else:
        xml_source = xml_directory

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    filtered_images = []
    # Select the inference images without xml
    for filename in images:
        print("Processing ", filename)
        image_path = os.path.join(source, filename)
        #xml_filename = os.path.join(source, os.path.splitext(filename)[0]+'.xml')
        xml_filename = os.path.join(xml_source, os.path.splitext(filename)[0] + '.xml')

        if not os.path.isfile(xml_filename):
            copyfile(image_path, os.path.join(inference_dir, filename))
            print("Added image {} to inference as no XML could be found.".format(filename))

            if remove_source:
                try:
                    os.remove(image_path)
                except OSError as e:  ## if failed, report it back to the user ##
                    print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            #Remove the copied image from the list as it does not have any xml
            filtered_images.append(filename) #images.remove(images[i])

    images = filtered_images

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)

    # Select the test images
    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            if os.path.isfile(os.path.join(xml_source, xml_filename)):
                copyfile(os.path.join(xml_source, xml_filename),
                         os.path.join(test_dir,xml_filename))
            else:
                print("Warning: No xml file {} for test image {}".format(os.path.join(xml_source, xml_filename), os.path.join(source, filename)))

        if remove_source:
            try:
                os.remove(os.path.join(source, filename))
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))

        images.remove(images[idx])

    for filename in images:
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            if os.path.isfile(os.path.join(xml_source, xml_filename)):
                copyfile(os.path.join(xml_source, xml_filename),
                         os.path.join(train_dir, xml_filename))
            else:
                print("Warning: No xml file {} for training image {}".format(os.path.join(source, xml_filename),
                                                                    os.path.join(source, filename)))

        if remove_source:
            try:
                os.remove(os.path.join(source, filename))
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-d', '--xmlDir',
        help='Path to the folder where the xml bounding boxes are stored. If not specified, '
             'the image directory will be used.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    parser.add_argument(
        '-x', '--xml_copy',
        help='Set this flag if you want the xml annotation files to be processed and copied over.',
        action='store_true'
    )
    parser.add_argument('-m', '--remove_source',
                        help='Set this flag to remove the source images',
                        action='store_true')


    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.ratio, args.xml_copy, args.xmlDir, args.remove_source)


if __name__ == '__main__':
    main()