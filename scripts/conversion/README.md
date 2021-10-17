# Converters
The following converters are used to convert between machine learning data formats for models and datasets. Their purpose is to make the life easier for developers by providing
complete converters. However, the converters might be adapted for other purposes. All converters are either written by the devlopers of the repository or parts are used from
other open source converters. Within each converter file, each source has been provided. For each converter, the goal is also to provide sample data as part of the converter 
documentation.

## Pre-requisites
Tensorflow object detection api 2.4 is necessary

## VOC to Coco
Source: [https://github.com/yukkyo/voc2coco](https://github.com/yukkyo/voc2coco)

Script: `convert_voc_to_coco.py` 

Example: 
```shell
python convert_voc_to_coco.py ^
--ann_dir samples/annotations/xml ^
--ann_ids samples/annotations/train.txt ^
--labels samples/annotations/labels.txt ^
--output samples/annotations/coco_train_annotations.json ^
--ext xml
```

Notes: 
- ann_dir: XML directory
- ann_ids: txt file with filenames without extensions. If this file is None, all files of ann_dir are used.
- labels: labels file 
- output: Output JSON
- ext: File extension to filter

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

Script modified: `convert_coco_to_tfrecord.py`

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

## VOC to Yololike format CustomText
Source: [https://github.com/david8862/keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set)

Data annotation file format:
One row for one image in annotation file;
Row format: image_file_path box1 box2 ... boxN;
Box format: x_min,y_min,x_max,y_max,class_id (no space).
Example: path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3

Script: `convert_voc_to_customtext.py` 

Example: 
```shell
echo Convert training data
set TYPE=train
python %SCRIPTPREFIX%\conversion\convert_voc_to_customtext.py ^
--annotations_dir=annotations/xmls ^
--image_dir=images/%TYPE% ^
--output_path=annotations/yolo/yolo_%TYPE%.txt ^
--classes_path=annotations/labels.txt ^
--include_difficult ^
--include_no_obj
```

## TF CSV to PASCAL VOC
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

## TF CSV to COCO JSON Detections
Convert Tensorflow CSV detection format to the Coco JSON detection format.

Source: 

Script: `convert_tfcsv_to_pycocodetections.py`

Examples:
```shell
python %SCRIPTPREFIX%\conversion\convert_tfcsv_to_pycocodetections.py ^
--annotation_file="results/%MODELNAME%/validation_for_inference/detections.csv" ^
--output_file="results/%MODELNAME%/validation_for_inference/coco_pets_detection_annotations.json
```

## TF2 Keras to TF1 Frozen - OBSOLETE
The following script converts a .h5 model to TF1 frozen model. Forced conversion should not be used. Therefore, this conversion method is obsolete. Instead, use convert_kerash5_to_tf2.py

Source:

Script: `convert_tf2keras_to_tf1frozen.py`

Examples:

## TF2 Keras to TF2 Saved Model
Source:

Script: `convert_kerash5_to_tf2.py`

Examples:
```
python convert_kerash5_to_tf2.py ^
--input_path="exported-models/tf2ke_yolo3mobilenetlite_448x448_pets/saved_model.h5" ^
--output_dir="exported-models/tf2ke_yolo3mobilenetlite_448x448_pets"
```


## Yolo to VOC
Convert Yolo annotations to Pascal VOC.

Source: https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd

Script: `convert_yolo_to_voc.py`

Arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  -ad ANNOTATION_DIR, --annotation_dir ANNOTATION_DIR
                        Annotation directory with txt files of yolo annotations of the same name format as image files
  -id IMAGE_DIR, --image_dir IMAGE_DIR
                        Image file directory
  -at TARGET_ANNOTATION_DIR, --target_annotation_dir TARGET_ANNOTATION_DIR
                        Target directory for xml files
  -cl CLASS_FILE, --class_file CLASS_FILE
                        File with class labels
  --create_empty_images
                        Generates xmls also for images without any found objects, i.e. empty annotations. It is useful to prevent overfitting.
```

Example:
```
python %SCRIPTPREFIX%\conversion\convert_yolo_to_voc.py ^
--annotation_dir "./annotations/yolo_labels" ^
--target_annotation_dir "./annotations/voc_from_yolo_labels" ^
--image_dir "images/train" ^
--class_file "./annotations/labels.txt" ^
--create_empty_images
```


## Yolo Detections to Tensorflow Detections CSV File
Convert Yolo detection annotations to Tensorflow detection annotations as csv. It is used to get all detection formats to fit the common detection format, i.e. tensorflow detection csv.

Source: -

Script: `convert_yolo_to_tfcsv.py`

Arguments:
```
optional arguments:
  -h, --help            show this help message and exit
parser.add_argument("-ad", '--annotation_dir',
                    default=None,
                    help='Annotation directory with txt files of yolo annotations of the same name format as image files',
                    required=False)
parser.add_argument("-id", '--image_dir',
                    default="images",
                    help='Image file directory to get the image size from the corresponding image', required=False)
parser.add_argument("-out", '--output',
                    default="./detections.csv",
                    help='Output file path for the detections csv.', required=False)
```

Example:
```
python %SCRIPTPREFIX%\conversion\convert_yolo_to_tfcsv.py ^
--annotation_dir=results/pt_yolov5s_640x360_oxfordpets_e300/TeslaV100/labels ^
--image_dir=images/validation ^
--output=results/pt_yolov5s_640x360_oxfordpets_e300/TeslaV100/detections.csv
```


## Darkent2caffe
Source: [Darknet](https://github.com/ysh329/darknet2caffe)

## VOC or COCO to Yolo
The conversion from VOC or Coco to yolo is added to this repository as a subrepository.
Source: [https://github.com/paulxiong/convert2Yolo/tree/8de035a4a003dcf6b5f383e8262ae4856646978c](https://github.com/paulxiong/convert2Yolo/tree/8de035a4a003dcf6b5f383e8262ae4856646978c)

Script: - 

Example: 
```shell
python C:\Projekte\21_SoC_EML\convert2Yolo\example.py ^
--datasets VOC ^
--img_path C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/images/train ^
--label C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/xmls ^
--convert_output_path C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/yolo_labels ^
--img_type ".jpg" ^
--manifest_path C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/ ^
--cls_list_file C:/Projekte/21_SoC_EML/scripts-and-guides-samples/oxford_pets_reduced/annotations/labels.txt
```

Notes: 
- see original repo for guide how to use the converter 
- makedirs is not used. Therefore, folders like yolo_labels have to be created manually

## png to jpg
Convert png images to jpg to keep a uniform format

Source: -

Script: `convert_png_to_jpg.py`

Arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  -id IMAGE_DIR, --image_dir IMAGE_DIR
                        Image file directory
```

Example:
```
python %SCRIPTPREFIX%\conversion\convert_png_to_jpg.py ^
--image_dir "images/train"
```

# Issues
If there are any issues or suggestions for improvements, please add an issue to github's bug tracking system or please send a mail 
to [Alexander Wendt](mailto:alexander.wendt@tuwien.ac.at)

<div align="center">
  <img src="../../_img/eml_logo_and_text.png", width="500">
</div>
