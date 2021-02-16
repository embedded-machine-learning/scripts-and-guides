# Converters
The following converters are used to convert between machine learning data formats for models and datasets. Their purpose is to make the life easier for developers by providing
complete converters. However, the converters might be adapted for other purposes. All converters are either written by the devlopers of the repository or parts are used from
other open source converters. Within each converter file, each source has been provided. For each converter, the goal is also to provide sample data as part of the converter 
documentation.

## Pre-requisites
Tensorflow object detection api 2.0 is necessary

## VOC to Coco
Source: [https://github.com/yukkyo/voc2coco](https://github.com/yukkyo/voc2coco)

Script: `convert_voc_to_coco.py` 

Example: 
```shell
python convert_voc_to_coco.py --ann_dir samples/annotations/xml --ann_ids samples/annotations/train.txt --labels samples/annotations/labels.txt --output samples/annotations/coco_train_annotations.json --ext xml
```

## Coco to VOC
Source: [https://gist.github.com/jinyu121](https://gist.github.com/jinyu121/a222492405890ce912e95d8fb5367977)

Script: `convert_coco_to_voc.py` 

Example: 
```shell
python convert_coco_to_voc.py --annotation_file="samples/annotations/cvml_xml/annotations/cvml_Milan-PETS09-S2L1_coco.json"
```

## CVML to Coco
Source: [CVML Annotation — What it is and How to Convert it?](https://towardsai.net/p/deep-learning/cvml-annotation%e2%80%8a-%e2%80%8awhat-it-is-and-how-to-convert-it)

Script: `convert_cvml_to_coco.py` 

Example: 
```shell
python convert_cvml_to_coco.py --annotation_file="samples/annotations/cvml_xml/cvml_Milan-PETS09-S2L1.xml" --image_dir="samples/cvml_images" --label_name="predestrian"
```

## 3DMOT2015 Yololike format to Coco
Source: Inspired from [here](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter/blob/master/main.py)

Script: `convert_3DMOT2015_yololike_to_coco.py`

Notice: The format from which is connected looks like the yolo format, but is not the same. This format was used in MOT Challenge 2015. However. This script can 
easily be transformed into a yolo converter.

Example: 
```shell
python C:\Projekte\21_SoC_EML\public_content\scripts-and-guides\scripts\conversion\convert_3DMOT2015_yololike_to_coco.py ^
--annotation_file="samples/annotations/3DMOT2015_yololike_ground_truth.txt" ^
--image_dir="samples/yolo_images" ^
--label_name="pedestrian" ^
--image_name_prefix=""
```

## Coco to TFRecords
Source: [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py)

Script original: `convert_coco_to_tfrecord.py`

Script modified: `convert_coco_to_tfrecord_mod.py`

Examples:
```shell
python convert_coco_to_tfrecord_mod.py --logtostderr --image_dir="samples/images" --annotations_file="samples/annotations/coco_train_annotations.json" --output_path="samples/prepared-records/train_coco.record" --number_shards=2
```

## VOC to TFRecords
Source: [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools)

Script original: 

Script modified: `convert_voc_to_tfrecord_mod.py`

Examples:
```shell
python convert_voc_to_tfrecord_mod.py -x "samples/annotations/xml" -i "samples/images" -l "samples/annotations/sw_label_map.pbtxt" -o "samples/prepared-records/train_voc.record" -n 2
```

## TF2 CSV to PASCAL VOC
Tensorflow uses a csv format that looks like the yolo format for intermediate representations. This script converts from the intermediate format to Pascal VOC.

Source: This script was inspired by [Shubham Gupta](https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd)

Script: `convert_tfcsv_to_voc.py`

Examples:
```shell
python %SCRIPTPREFIX%\conversion\convert_tfcsv_to_voc.py ^
--annotation_file="results/tf2oda_efficientdetd2_768_576_coco17_pedestrian_all/detections.csv" ^
--output_dir="results/tf2oda_efficientdetd2_768_576_coco17_pedestrian_all/xml" ^
--labelmap_file="annotations/pedestrian_label_map.pbtxt"
```

## TF2 Keras to TF1 Frozen
Source:

Script: `convert_tf2keras_to_tf1frozen.py`

Examples:

## Darkent2caffe
Source: [Darknet](https://github.com/ysh329/darknet2caffe)

# Issues
If there are any issues or suggestions for improvements, please add an issue to github's bug tracking system or please send a mail 
to [Alexander Wendt](mailto:alexander.wendt@tuwien.ac.at)

<div align="center">
  <img src="../../_img/eml_logo_and_text.png", width="500">
</div>
