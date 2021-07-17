#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in/Generic Imports
import glob
import json
import os

# Libs
from tqdm import tqdm
from xmltodict import unparse
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(
    description="Remove invalid bounding boxes from xml annotations files"
)
parser.add_argument(
    "-af",
    "--annotation_folder",
    default="samples/tmp",
    help="Annotation folder.",
    required=False,
)

parser.add_argument(
    "-o",
    "--output_folder",
    default="samples/tmp/renamed",
    help="Output folder for renamed files",
    required=False,
)

args = parser.parse_args()


def remove_invalid_bounding_boxes_in_xml(annotation_folder, output_folder):

    # Output folder generation
    if output_folder != annotation_folder:
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            print("Created ", output_folder)
    else:
        print("output_folder is the same as the input folder. Replace files")

    # xml_file_list = []
    print("Filter not used. Select all xml files of the folder")
    xml_file_list = glob.glob(annotation_folder + "/*.xml")
    for xml_file in xml_file_list:  # glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        ## Get size of image
        image_width = int(root.find("size")[0].text)
        image_height = int(root.find("size")[1].text)

        ## Iterating over all children to find objects and then checking if the bounding box coordinates are out of rannge
        for object in root.findall("object"):
            if len(list(object))>0:
                xmin = int(object.find("bndbox").find("xmin").text)
                ymin = int(object.find("bndbox").find("ymin").text)
                xmax = int(object.find("bndbox").find("xmax").text)
                ymax = int(object.find("bndbox").find("ymax").text)
                if (
                    (xmin < 0)
                    or (xmax > image_width)
                    or (ymin < 0)
                    or (ymax > image_height)
                    or (xmin > xmax)
                    or (ymin > ymax)
                ):
                    print("in {}, size {}x{}, remove bbox xmin {}, ymin {}, xmax {}, ymax {}".
                          format(xml_file, image_width, image_height, xmin, ymin, xmax, ymax))
                    root.remove(object)

        # Save xml again
        target_path = os.path.join(output_folder, os.path.basename(xml_file))
        tree.write(target_path)
        print("File saved in {}".format(target_path))


if __name__ == "__main__":

    remove_invalid_bounding_boxes_in_xml(args.annotation_folder, args.output_folder)

    print("=== Program end ===")