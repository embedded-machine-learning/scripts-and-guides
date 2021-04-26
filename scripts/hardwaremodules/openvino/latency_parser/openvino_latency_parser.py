# command to generate report with
# python3 /opt/intel/openvino_2020.4.287/deployment_tools/tools/benchmark_tool/benchmark_app.py \
# --path_to_model ~/projects/nuc_format/inference-demo/exported-models/tf2oda_efficientdetd0_512x384_pedestrian_LR02/saved_model/saved_model.xml \
# --report_type average_counters -niter 10 -d MYRIAD -nireq 1

# Data Formats
# Date Model Model_Short Framework Network Resolution Dataset Custom_Parameters Hardware Hardware_Optimization DetectionBoxes_Precision/mAP DetectionBoxes_Precision/mAP@.50IOU DetectionBoxes_Precision/mAP@.75IOU DetectionBoxes_Precision/mAP (small) DetectionBoxes_Precision/mAP (medium) DetectionBoxes_Precision/mAP (large) DetectionBoxes_Recall/AR@1 DetectionBoxes_Recall/AR@10 DetectionBoxes_Recall/AR@100 DetectionBoxes_Recall/AR@100 (small) DetectionBoxes_Recall/AR@100 (medium) DetectionBoxes_Recall/AR@100 (large)
# Date Model Model_Short Framework Network Resolution Dataset Custom_Parameters Hardware Hardware_Optimization Batch_Size Throughput Mean_Latency Latencies

# EXAMPLE USAGE - the following command extracts infos from reports and parses them into a new file
# python3 openvino_latency_parser.py --avg_rep tf_inceptionv1_224x224_imagenet_3.16G_avg_cnt_rep.csv --inf_rep tf_inceptionv1_224x224_imagenet_3.16G.csv --save_new

#EXAMPLE USAGE - the following command extracts infos from reports and appends them to a new line of the existing_file csv
# python3 openvino_latency_parser.py --avg_rep tf_inceptionv1_224x224_imagenet_3.16G_avg_cnt_rep.csv --inf_rep tf_inceptionv1_224x224_imagenet_3.16G.csv --existing_file latency_tf_inceptionv1_224x224_imagenet_3.16G.csv

import os, sys, csv, argparse
from datetime import datetime

# keywords used in the Openvino reports
# extract from info rep: name, hardware, batch, mode (sync,async), throughput, latency
keywords = ["target device", "--path_to_model", "number of parallel infer requests",
                        "API", "batch size", "latency (ms)", "throughput"]#, "precision"]
# keywords used in the EML data structure
latency_keywords = ["Date", "Model", "Model_Short",	"Framework", "Network", "Resolution", "Dataset", "Custom_Parameters",
                    "Hardware", "Hardware_Optimization", "Batch_Size", "Throughput", "Mean_Latency", "Latencies"]


def file_exists(file_path):
    """Simple function checks if file exists at file_path

    :param file_path: path to file to check
    :return: True if file exists, False if it doesn't
    """
    if not os.path.isfile(file_path):
        print("Could not find file under:", file_path)
        return False
    return True


def read_csv_report(datafile_path, delimiter):
    """Parses data from csv file to list and returns it

    :param datafile_path: path to report file
    :param delimiter: delimiter used in the report file
    :return: list with data from csv or None when unable to parse data
    """
    with open(datafile_path, "r", newline='') as report_file:
        if delimiter == None:
            delimiter = ";"
        print("using ", delimiter, " as delimiter")
        return list(csv.reader(report_file, delimiter=delimiter, quotechar='|'))
    return None


def extract_information_avg_rep(report_data):
    # structure of data: ['layerName', 'execStatus', 'layerType', 'execType', 'realTime (ms)', 'cpuTime (ms)']

    # go over the data, cast it to float at append to a list
    layers_durations = []
    layer_names = []
    network_duration = 0
    for row in report_data:
        if row == []:
            break  # if line is empty, end loop
        # print(row, len(row))
        # check if the rows contains valid data in format xxx.xxx where the . signifies a float value
        if len(row) > 3 and "." in row[4]:
            if float(row[4]) > 0:
                #print(row[0], float(row[4]))
                layer_names.append(row[0])
                layers_durations.append(float(row[4]))
            if row[0] == "Total":
                network_duration = float(row[4])
    #print("\nnetwork duration: {0:5f} ms\n".format(network_duration))

    return layers_durations, layer_names, network_duration


def extract_information_inf_rep(report_data):
    # inf rep data format is ["Command line parameters", parameters]
    extracted_inf = {} # prepare a dictionary where to put the extracted information into
    for row in report_data:
        if row == []:
            continue  # if line is empty, skip row

        if len(row) > 0 and row[0] in keywords:
            print(row)
            try:
                if row[0] == "--path_to_model":
                    # augment the data - more information can be extracted from model file path
                    # e.g. /home/matvey/projects/models/xilinx_models/xilinx_model_movid/tf2oda_ssdmobilenetv2_300x300_pedestrian_D100_LR08.csv
                    extracted_inf["--path_to_model"] = row[1]
                    if "/" in extracted_inf["--path_to_model"]:
                        if ".xml" in extracted_inf["--path_to_model"]:
                            extracted_inf["full_name"] = extracted_inf["--path_to_model"].split("/")[-1].split(".xml")[0] # get full name from path to model
                            if "_" in extracted_inf["full_name"]:
                                try:
                                    extracted_inf["short_name"] = extracted_inf["full_name"].split("_")[1]
                                    extracted_inf["framework"] = extracted_inf["full_name"].split("_")[0]
                                    extracted_inf["resolution"] = extracted_inf["full_name"].split("_")[2]
                                    extracted_inf["custom_params"] = extracted_inf["full_name"].split("_")[4:]
                                except:
                                    print("Could not split ", extracted_inf["full_name"], " to extract data, because a '_' is missing. " \
                                                                          "is the format correct?")
                            else:
                                extracted_inf["short_name"] = None
                                extracted_inf["framework"] = None
                                extracted_inf["resolution"] = None
                                extracted_inf["custom_params"] = None
                        else:
                            extracted_inf["full_name"] = None
                    else:
                        print("No ’/’ in ", extracted_inf["full_name"], ". Please check if parsed information is correct.")
                    continue
                if row[0] == "target device" and row[1] == "MYRIAD":
                    extracted_inf["precision"] = "FP16"

                extracted_inf[row[0]] = row[1] # add data to dict
                print("filling in",row[1])
            except:
                continue # if row[0] not in keywords, skip row

    return extracted_inf


def reformat_inf(extracted_inf):
    # build a dataframe according to the latency data format in latency_keywords
    new_frame = [str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))] # Date
    new_frame.append(extracted_inf["full_name"])                    # Model
    new_frame.append(extracted_inf["short_name"])                   # Model_Short
    new_frame.append(extracted_inf["framework"])                    # Framework
    new_frame.append(extracted_inf["short_name"])                   # Network
    new_frame.append(extracted_inf["resolution"])                   # Resolution
    new_frame.append(None)                                          # Dataset
    new_frame.append(list((extracted_inf["number of parallel infer requests"], extracted_inf["API"], extracted_inf["precision"], extracted_inf["custom_params"]))) # Custom_Parameters
    new_frame.append(extracted_inf["target device"])                # Hardware
    new_frame.append(None)                                          # Hardware_Optimization
    new_frame.append(extracted_inf["batch size"])                   # Batch_Size
    new_frame.append(extracted_inf["throughput"])                   # Throughput
    new_frame.append(extracted_inf["latency (ms)"])                 # Mean_Latency
    new_frame.append([None])                                        # Latencies
    print(new_frame)
    return new_frame


def parse_avg_report(report_path, delimiter):
    # check if passed report file exists
    if not file_exists(report_path):
        return

    # get the data from file assuming an Openvino average counters benchmark report
    report_data = read_csv_report(report_path, delimiter)
    layers_durations, layer_names, network_duration = extract_information_avg_rep(report_data)

    return layers_durations, layer_names, network_duration


def parse_inf_report(report_path, delimiter):
    # check if passed report file exists
    if not file_exists(report_path):
        return

    # get the data from file assuming an  openvino benchmark app information report

    report_data = read_csv_report(report_path, delimiter)
    extracted_inf = extract_information_inf_rep(report_data)

    return extracted_inf


def save_new_rep(extracted_inf, reformated_inf):
    #print(extracted_inf)
    # create a new file handle for new report
    with open("latency_" + extracted_inf["full_name"] + ".csv", "w") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(latency_keywords) # write the upper column according to dataframe
        writer.writerow(reformated_inf)

        print("Writing file successful!")


def append_to_file(existing_file, extracted_inf):
    if not file_exists(existing_file):
        print("Parsed file on which to append data ", existing_file, " does not exist.")
        return
    # open existing file and append extracted information
    with open(existing_file, "a+") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(reformated_inf)

        print("Writing to " + existing_file + " successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCS2 settings test')
    parser.add_argument("-ar", '--avg_rep', type=str,
                        help='report containing average counters from the Openvino benchmark app', required=False)
    parser.add_argument("-ir", '--inf_rep', type=str,
                        help='report containing execution information from the Openvino benchmark app', required=False)
    parser.add_argument("-e", '--existing_file', type=str,
                        help='a file in the desired format, where newly extracted data shall be appended', required=False)
    parser.add_argument("-l", '--delimiter', type=str, default=";",
                        help='whatever the csv uses as delimiter', required=False)

    parser.add_argument('--save_new', dest='save_new', action='store_true')
    parser.add_argument('--no-save_new', dest='save_new', action='store_false')
    parser.set_defaults(save=False)

    args = parser.parse_args()

    if args.avg_rep:
        print("Parsing the average counters report...")
        layers_durations, layer_names, network_duration = parse_avg_report(args.avg_rep, args.delimiter)
        print("\nlayers_durations\n", layers_durations, "\nlayer_names\n",layer_names,"\nnetwork_duration\n", network_duration)
    else:
        print("Skipping the average counters report...")

    if args.inf_rep:
        print("Parsing the inference information report...")
        extracted_inf = parse_inf_report(args.inf_rep, args.delimiter)
        print("\nextracted_inf\n", extracted_inf)
        reformated_inf = reformat_inf(extracted_inf)
        print("\nreformated_inf\n", reformated_inf)
    else:
        print("Skipping the inference information report...")

    if args.save_new:
        print("Saving extracted information to new file...")
        save_new_rep(extracted_inf, reformated_inf)
    elif args.existing_file:
        # append data to existing file
        append_to_file(args.existing_file, reformated_inf)
    else:
        print("No flags to save or append to file were given.")