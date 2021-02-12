import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os


#Source: https://www.programmersought.com/article/2053478686/

def save_xml(image_name, bbox, save_dir='./VOC2007/Annotations', width=1609, height=500, channel=3):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for x, y, x1, y1 in bbox:
        left, top, right, bottom = x, y, x1, y1
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'person'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


def change2xml(label_dict={}):
    for image in label_dict.keys():
        image_name = os.path.split(image)[-1]
        bbox = label_dict.get(image, [])
        save_xml(image_name, bbox)
    return


import pandas as pd
import numpy as np

data = pd.read_table("path/train_labels.csv", sep=",")
name_file = open('path/ImageSets/Main/train.txt', 'r')
name_file = name_file.readlines()
for name in name_file:
    img = cv2.imread('path/JPEGImages/' + name[:-1] + '.jpg')
    height, width = img.shape[:2]
    name = name[:-1] + '.jpg'
    xx = np.array(data[data['ID'] == name][' Detection'])
    bbox = []
    for i in range(xx.shape[0]):
        bbox.append(xx[i].split(' '))
    save_xml(image_name=name, bbox=bbox, save_dir='path/Annotations', width=width, height=height, channel=3)