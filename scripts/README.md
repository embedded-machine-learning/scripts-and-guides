# Embedded Machine Learning Toolbox

The EML Toolbox is a collection of scripts by the <a href="https://embedded-machine-learning.github.io/webpage/">Christian Doppler Laboratory for Embedded Machine Learning</a> 
for automated and simplified training and inference of neural networks. We collect and unify scripts that help us in our everyday work to setup software and frameworks.

The purpose of using unified scripts on different hardware platforms is to solve the following task: Considering requirements regarding latency, accuracy and environmental factors,
we want to find the best combination of a 
- Neural network
- Neural network optimization
- Hardware 
- Hardware configuration

Secondary, we want to estimate or measure the effect of a particular network on multiple hardware to see where it works well.

## Overview

### EML Toolbox Scripts
* [ Converters](./conversion)
* [ Data Processing Tools](./data_preparation)
* [ Hardware Modules](./hardwaremodules/)
* [ Inference Evaluation Tools](./inference_evaluation)
* [ Power Measurements](./power_measurements)
* [ Training Tools](./training)
* [ Visualization] (./visualization)

### Additional Toolbox Extras
* [ Template Folder Structure for Tensor Flow](https://github.com/embedded-machine-learning/scripts-and-guides-templates)
* [ Sample Projects](https://github.com/embedded-machine-learning/scripts-and-guides-samples)

## Architecture

The architecture conists of several parts:
- Toolbox: It provides the infrastructure through general python scripts for data preparation, training, inference and evaluation. Further, it provides common interfaces 
for the hardware inference engines and model optimizers. The interfaces are then utilized by execution scripts. 
- Network optimizers: Plug-in Pruning- and Quantization Tools that can be applied to a network to optimize its performance for a certain hardware device
- Network estimators: In a network search, estimators provide a possibility to test a certain hardware without having to implement the network on it. It saves engineering effort and
makes the process faster
- Hardware: The embedded hardware devices, which are connected to the toolbox and available for network tests
- Hardware configuration optimizers: For each hardware, there is the possibility to setup the hardware for minimum latency, minimum power or minimum energy consumption 

<div align="center">
  <img src="./_img/emltoolbox_architecture.png", width="500">
</div>


## Implementation
At the current state, the EML toolbox supports the following hardware platforms:
- NVIDIA Xavier
- Intel NUC

It support the following networks:
- Tensorflow 2 Object Detection API SSD-MobileNet
- Tensorflow 2 Object Detection API EfficientDet

## Interfaces and Conventions

### Hardware Module Interfaces
Hardware module interfaces:
- Latency
- Performance for object detection

### Network File Names
Much information is put into the filename of a certain network. Many evaluation tools use this information from the position in the file name. Therefore, it is important to
keep on to this conventions, in order to prevent cusomization of tools.

Network file name convention:
[FRAMEWORK]_[NETWORKNAME]_[RESOLUTION_X]x[RESOLUTION_Y]_[DATASET]_[CUSTOM_PARAMETER_1]_[CUSTOM_PARAMETER_2]..._[CUSTOM_PARAMETER_n]

[FRAMEWORK]: 
- cf: Caffe
- tf2: Tensorflow 2
- tf2oda: Tensorflow 2 Object Detection API 
- dk: Darknet
- pt: PyTorch

If no dataset is known, the following syntax is used.
[DATASET] unknown: "ND"

Examples:
- tf2_mobilenetV2_224x224_coco_D100
- pt_refinedet_480x360_imagenet_LR03_WR04
- tf2oda_ssdmobilenetv2_320x320_pedestrian




## Guides how to use the Toolbox

### Setup Folder Structure for Training

### Setup Folder Structure for Inference

### Execute Inference on Intel Hardware

### Execute Inference on NVIDIA Hardware

## Upcoming

### Hardware Platforms
- Edge TPU USB Stick

### Networks
- YoloV4




#### Usage of the Scripts Repository
- Create a folder for this setup.
- Clone the scripts and guides repository into that folder, i.e. \[basefolder\]/scripts-and-guides
- Create a folder projects i.e. \[basefolder\]/projects
- Create your on project from the templates in scripts-and-guides in the projects folder like \[basefolder\]/projects/my_eml_project
- Copy all necessary scripts into the my_eml_project folder and adjust the paths. 

<div align="center">
  <img src="./../_img/eml_logo_and_text.png", width="500">
</div>
