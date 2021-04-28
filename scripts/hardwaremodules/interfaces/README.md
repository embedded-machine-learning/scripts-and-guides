# Hardware Module Interfaces

The idea of the EML Toolbox is to be able to apply the same folder structure and scripts on different hardware to save a huge amount of customization and engineering effort.
To make it possible, each hardware with its native model optimizer and inference executor have to use a common interface for data exchange with the EML Toolbox. In this folder,
we added example interfaces to the hardware modules

The basic idea is to exchange data through csv files, which are very general. Each measurement set shall be added as one row in the file. For each interface certain column 
headers are defined. These headers are the interfaces, which shall be common. It is possible to merge latency and performance into one file with multiple or even sparse use of 
headers

## Latency Measurement Interface
The following column headers shall be used for latency exchange:
- Date: Format yyyy-MM-DD HH:mm:ss e.g. 2021-03-09  23:29:00
- Model: Long model name as used as file name, e.g. tf2oda_ssdmobilenetv2_320x320_pedestrian
- Model_Short	
- Framework	
- Network	
- Resolution	
- Dataset	
- Custom_Parameters	
- Hardware	
- Hardware_Optimization	
- Batch_Size	
- Throughput	
- Mean_Latency	
- Latencies



In [ hw_module_example_latency.csv](./hw_module_example_latency.csv), we present an example interface

## Performance Measurement Interface for Object Detection



## Upcoming
We plan to create a database, where measurements are automatically inserted according to the interfaces above.


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


## Guides how to use it

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
