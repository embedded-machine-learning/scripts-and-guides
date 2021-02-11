#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert COCO annotations to Pascal VOC.

License_info:
# ==============================================================================
# ISC License (ISC)
# Copyright 2021 Christian Doppler Laboratory for Embedded Machine Learning
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

# The following is a slightly modified version from the following script
# Source: https://gist.github.com/jinyu121/a222492405890ce912e95d8fb5367977

"""

# Futures
#from __future__ import print_function

# Built-in/Generic Imports
import json
import os

# Libs
from tqdm import tqdm
from xmltodict import unparse
import argparse

# Own modules

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.5.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

parser = argparse.ArgumentParser(description='Convert Coco to VOC')
#parser.add_argument("-ad", '--annotation_dir',
#                    default=None,
#                    help='Annotation directory with xml files of the same format as image files', required=False)
parser.add_argument("-af", '--annotation_file',
                    default="samples/annotations/cvml_xml/annotations/cvml_Milan-PETS09-S2L1_coco.json",
                    help='Annotation file.', required=False)
parser.add_argument("-b", '--bbox_offset', type=int,
                    default=0,
                    help='BOX_OFFSET: Switch between 0-based and 1-based bbox. The COCO dataset is in 0-based format, '
                         'while the VOC dataset is 1-based. To keep 0-based, set it to 0. To convert to 1-based, '
                         'set it to 1.', required=False)
#parser.add_argument("-id", '--image_dir',
#                    default="samples/cvml_images",
#                    help='Image file directory for writing the trainval.txt file', required=False)
#parser.add_argument("-label", '--label_name',
#                    default="pedestrian",
#                    help='Label of the set in a binary set. Default is pedestrian.', required=False)
#parser.add_argument("-del", '--delete_csv', default="True", help='Delete csv', required=False)

args = parser.parse_args()


def base_dict(filename, width, height, depth=3):
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            "folder": "VOCCOCO", "segmented": "0", "owner": {"name": "unknown"},
            "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
            "size": {'width': width, 'height': height, "depth": depth},
            "object": []
        }
    }


def base_object(size_info, name, bbox, bbox_offset=0):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = max(x1, 0) + bbox_offset
    y1 = max(y1, 0) + bbox_offset
    x2 = min(x2, width - 1) + bbox_offset
    y2 = min(y2, height - 1) + bbox_offset

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }

def convert_coco_to_voc(coco_json_path, bbox_offset=0):
    '''

    Parameters
    ----------
    coco_json_path: Path of coco json file



    Returns
    -------

    '''

    # BBOX_OFFSET: Switch between 0-based and 1-based bbox.
    # The COCO dataset is in 0-based format, while the VOC dataset is 1-based.
    # To keep 0-based, set it to 0. To convert to 1-based, set it to 1.
    BBOX_OFFSET = 0

    src_base = coco_json_path #"samples/annotations/Milan-PETS09-S2L1.xml.csv_coco.json" #os.path.join("data", "coco", "annotations")
    dst_base = os.path.dirname(src_base) #os.path.join(os.path.dirname(src_base), "coco2voc_xml")

    dst_dirs = {x: os.path.join(dst_base, x) for x in ["annotations", "imagesets", "jpgimages"]}
    dst_dirs['imagesets'] = os.path.join(dst_dirs['imagesets'], "Main")
    for k, d in dst_dirs.items():
        os.makedirs(d, exist_ok=True)


    sets = {
        "trainval": src_base, #os.path.join(src_base, "instances_train2017.json"),
        #"test": src_base,# os.path.join(src_base, "instances_val2017.json"),
    }

    cate = {x['id']: x['name'] for x in json.load(open(sets["trainval"]))['categories']}

    for stage, filename in sets.items():
        print("Parse", filename)
        data = json.load(open(filename))

        images = {}
        for im in tqdm(data["images"], "Parse Images"):
            if not 'coco_url' in im:
                im['coco_url'] = ""
            img = base_dict(im['coco_url'], im['width'], im['height'], 3)
            images[im["id"]] = img

        for an in tqdm(data["annotations"], "Parse Annotations"):
            ann = base_object(images[an['image_id']]['annotation']["size"], cate[an['category_id']], an['bbox'], bbox_offset)
            images[an['image_id']]['annotation']['object'].append(ann)

        image_name_bases = []
        for k, im in tqdm(images.items(), "Write Annotations"):
            im['annotation']['object'] = im['annotation']['object'] or [None]
            #Define files name
            #xml_filename = str(k).zfill(12)
            image_name = data['images'][k]['filename']
            xml_filename_base = image_name.split('.')[0]
            image_name_bases.append(xml_filename_base)

            im['annotation']['filename'] = image_name

            unparse(im,
                    open(os.path.join(dst_dirs["annotations"], "{}.xml".format(xml_filename_base)), "w"),
                    full_document=False, pretty=True)

        #Additionally write the trainvar for the images
        print("Write image sets")
        with open(os.path.join(dst_dirs["imagesets"], "{}.txt".format(stage)), "w") as f:
            #f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))
            f.writelines(list(map(lambda x: str(x) + "\n", image_name_bases)))

        print("OK")

if __name__ == "__main__":

    convert_coco_to_voc(args.annotation_file, args.bbox_offset)


    print("=== Program end ===")
