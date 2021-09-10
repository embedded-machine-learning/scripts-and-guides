#!/bin/bash

# put in execution folder

echo "Run complete execution process: convert model, execute converted model, execute TF2 model"

./add_folder_conv_ir.sh
./add_folder_infopenvino_jobs.sh
./add_folder_inftf2_jobs.sh