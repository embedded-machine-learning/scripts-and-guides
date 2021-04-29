# Hardware Module Interfaces

The idea of the EML Toolbox is to be able to apply the same folder structure and scripts on different hardware to save a huge amount of customization and engineering effort.
To make it possible, each hardware with its native model optimizer and inference executor have to use a common interface for data exchange with the EML Toolbox. In this folder,
we added example interfaces to the hardware modules

The basic idea is to exchange data through csv files, which are very general. Each measurement set shall be added as one row in the file. For each interface certain column 
headers are defined. These headers are the interfaces, which shall be common. It is possible to merge latency and performance into one file with multiple or even sparse use of 
headers

## Latency Measurement Interface
The following column headers shall be used for latency exchange:
- Id: Identifier for the measurement set, e.g. a hash like HJDSJD83. The id is used to merge and enhance measurement sets to get everything as a single entry in a database
- Date: Format yyyy-MM-DD HH:mm:ss e.g. 2021-03-09  23:29:00
- Model: Long model name as used as file name, e.g. tf2oda_ssdmobilenetv2_320x320_pedestrian
- Model_Short: For usage in figures, a short version of the model should be used, e.g. ssdmobnet_320x320. In case no special short name is supposed to be used, the long name should be used.
- Framework: network framework, e.g. tf2oda	
- Network: network name as one word, e.g. ssdmobilenetv2
- Resolution: Width x height, e.g. 320x300
- Dataset: Dataset name, e.g. pedestrian
- Custom_Parameters: The last parameters are custom parameters like LR03 for learning rate 0.3
- Hardware: Hardware name, e.g. Xavier
- Hardware_Optimization: Hardware optimization framework and settings like trt_int8
- Batch_Size: default 1
- Throughput: measured value
- Mean_Latency: calculated value from single measurements
- Latencies: single latency measurements

The network specific settings can be extracted from the long name according to the naming convention.

In [ hw_module_example_latency.csv](./hw_module_example_latency.csv), we present an example interface

## Performance Measurement Interface for Object Detection
The following column headers shall be used for performance of object detection exchange. The network+hardware identifier is 
- Id: Identifier for the measurement set, e.g. a hash like HJDSJD83. The id is used to merge and enhance measurement sets to get everything as a single entry in a database
- Date: Format yyyy-MM-DD HH:mm:ss e.g. 2021-03-09 23:29:00
- Model: Long model name as used as file name, e.g. tf2oda_ssdmobilenetv2_320x320_pedestrian
- Model_Short: For usage in figures, a short version of the model should be used, e.g. ssdmobnet_320x320. In case no special short name is supposed to be used, the long name should be used.
- Framework: network framework, e.g. tf2oda	
- Network: network name as one word, e.g. ssdmobilenetv2
- Resolution: Width x height, e.g. 320x300
- Dataset: Dataset name, e.g. pedestrian
- Custom_Parameters: The last parameters are custom parameters like LR03 for learning rate 0.3
- Hardware: Hardware name, e.g. Xavier
- Hardware_Optimization: Hardware optimization framework and settings like trt_int8
- DetectionBoxes_Precision/mAP: Part of the standard Coco metric
- DetectionBoxes_Precision/mAP@.50IOU: Part of the standard Coco metric
- DetectionBoxes_Precision/mAP@.75IOU: Part of the standard Coco metric
- DetectionBoxes_Precision/mAP (small): Part of the standard Coco metric
- DetectionBoxes_Precision/mAP (medium): Part of the standard Coco metric
- DetectionBoxes_Precision/mAP (large): Part of the standard Coco metric
- DetectionBoxes_Recall/AR@1: Part of the standard Coco metric
- DetectionBoxes_Recall/AR@10: Part of the standard Coco metric
- DetectionBoxes_Recall/AR@100: Part of the standard Coco metric
- DetectionBoxes_Recall/AR@100 (small): Part of the standard Coco metric
- DetectionBoxes_Recall/AR@100 (medium): Part of the standard Coco metric
- DetectionBoxes_Recall/AR@100 (large): Part of the standard Coco metric

Hint: In the programming of the interfaces, the id can be passed between the latency and the performance measurements via a temporary file identifier.tmp, which is deleted after the last usage in the execution script.

## Upcoming
- Create a merge script that creates only one csv file entry from both
- We plan to create a database, where measurements are automatically inserted according to the interfaces above.
- Add performance segmentation interface
- Add performance classification interface (no prio)

<div align="center">
  <img src="./../../../_img/eml_logo_and_text.png", width="500">
</div>
