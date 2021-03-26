# this script takes a neural network in the intermediate representation .pd
# and converts it to a Movidius NCS2 conform format with .xml and .bin
# runs inference on the generated model

import argparse
import os, sys, threading
from os import system
from sys import stdout
from time import sleep, time
import numpy as np
uldaq_import = False
try:
    from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag, ScanStatus,
                       ScanOption, create_float_buffer, InterfaceType, AiInputMode)
    print("Import of uldaq library for daq-card successful")
    uldaq_import = True
except:
    print("Could not load uldaq library for daq-card")

bench_over = False

def clear_eol():
    """Clear all characters to the end of the line."""
    stdout.write('\x1b[2K')


def daq_setup(rate=500000, samples_per_channel=500000*10, resistor=0.1):
    """Analog input scan example."""
    global bench_over
    bench_over = False # beginning of a new cycle
    daq_device = None
    status = ScanStatus.IDLE
    # samples_per_channel (int): the number of A/D samples to collect from each channel in the scan.
    # rate (float): A/D sample rate in samples per channel per second.

    range_index = 0
    interface_type = InterfaceType.ANY
    low_channel = 0
    high_channel = 0
    scan_options = ScanOption.CONTINUOUS
    flags = AInScanFlag.DEFAULT

    # Get descriptors for all of the available DAQ devices.
    devices = get_daq_device_inventory(interface_type)
    number_of_devices = len(devices)
    if number_of_devices == 0:
        raise RuntimeError('Error: No DAQ devices found')

    print('Found', number_of_devices, 'DAQ device(s):')
    for i in range(number_of_devices):
        print('  [', i, '] ', devices[i].product_name, ' (',
              devices[i].unique_id, ')', sep='')

    descriptor_index = 0
    if descriptor_index not in range(number_of_devices):
        raise RuntimeError('Error: Invalid descriptor index')

    # Create the DAQ device from the descriptor at the specified index.
    daq_device = DaqDevice(devices[descriptor_index])

    # Get the AiDevice object and verify that it is valid.
    ai_device = daq_device.get_ai_device()

    if ai_device is None:
        raise RuntimeError('Error: The DAQ device does not support analog '
                           'input')

    # Verify the specified device supports hardware pacing for analog input.
    ai_info = ai_device.get_info()

    if not ai_info.has_pacer():
        raise RuntimeError('\nError: The specified DAQ device does not '
                           'support hardware paced analog input')

    # Establish a connection to the DAQ device.
    descriptor = daq_device.get_descriptor()
    print('\nConnecting to', descriptor.dev_string, '- please wait...')
    # For Ethernet devices using a connection_code other than the default
    # value of zero, change the line below to enter the desired code.
    daq_device.connect(connection_code=0)

    # The default input mode is DIFFERENTIAL.
    input_mode = AiInputMode.DIFFERENTIAL

    # Get the number of channels and validate the high channel number.
    number_of_channels = ai_info.get_num_chans_by_mode(input_mode)

    if high_channel >= number_of_channels:
        high_channel = number_of_channels - 1
    channel_count = high_channel - low_channel + 1

    # Get a list of supported ranges and validate the range index.
    ranges = ai_info.get_ranges(input_mode)
    if range_index >= len(ranges):
        range_index = len(ranges) - 1
    meas_range = ranges[1]

    data_dir = "data_dir"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # define name of a datafile and delete if it exists in the data directory
    data_fname = "test_data"
    #if os.path.isfile(os.path.join(data_dir, data_fname)):
        #os.remove(os.path.join(data_dir, data_fname))

    # Allocate a buffer to receive the data.
    data = create_float_buffer(channel_count, samples_per_channel)
    # ranges[1] = Range.BIP5VOLTS

    return daq_device, low_channel, high_channel, input_mode, meas_range, samples_per_channel,  rate, scan_options, flags, data, data_dir, data_fname


def daq_measurement(daq_device, low_channel, high_channel, input_mode,
                    meas_range, samples_per_channel,
                    rate, scan_options, flags, data, data_dir, data_fname, index_run, api):
    # Start the acquisition.
    global bench_over
    ai_device = daq_device.get_ai_device()

    rate = ai_device.a_in_scan(low_channel, high_channel, input_mode,
                               meas_range, samples_per_channel,
                               rate, scan_options, flags, data)
    index = 0
    while not bench_over:
        # Get the status of the background operation
        status, transfer_status = ai_device.get_scan_status()
        # get current index
        index = transfer_status.current_index
        # when the index has reached maximal length

        #print("{} {}".format(samples_per_channel, index), end="\r", flush=True)

    # save data
    #print("jumped to break")
    with open(os.path.join(data_dir, "_".join((data_fname, str(index_run), api + ".dat"))), "wb") as f:
        np.save(f, np.asarray(data[0:index]))

    # Stop the acquisition if it is still running, disconnect
    if daq_device:
        if status == ScanStatus.RUNNING:
            ai_device.scan_stop()
        if daq_device.is_connected():
            daq_device.disconnect()
        daq_device.release()


def run_bench(daq_device, low_channel, high_channel, input_mode,ranges, samples_per_channel, rate, scan_options, flags,
              data, data_dir, data_fname,  power_measurement, index_pm,
              xml = "", pb = "",save_folder = "./tmp", report_dir = "report", niter = 100, api = "sync", proto=""):
    global bench_over
    mo_file = os.path.join("/", "opt", "intel", "openvino",
    "deployment_tools", "model_optimizer", "mo.py")
    bench_app_file = os.path.join("/","opt","intel", "openvino",
    "deployment_tools", "tools", "benchmark_tool", "benchmark_app.py")

    # check if necessary files exists
    if not os.path.isfile(mo_file):
        print("model optimizer not found at:", mo_file)

    if not os.path.isfile(bench_app_file):
        print("benchmark_app not found at:", bench_app_file)

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    print("\n**********Movidius FP16 conversion**********")
    xml_path = ""
    model_name = ""

    if not os.path.isfile(pb):
        if os.path.isfile(xml):
            xml_path = xml
            print("using already converted model! --> skipping conversion")
        else:
            sys.exit("Please enter a valid IR!")
    else:
        # yolov3/yolov3-tiny json file necessary for conversion
        conv_cmd_str = ""
        if "yolov3-tiny" in pb or "yolov3-tiny" in xml :
            conv_cmd_str = (" --tensorflow_use_custom_operations_config" +
            " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json")
        elif "yolov3" in pb or "yolov3-tiny" in xml :
            conv_cmd_str = (" --tensorflow_use_custom_operations_config" +
            " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json")

        if "tf_" in pb:
            # Tensorflow conversion
            # input_shape for tensorflow : batch, width, height, channels
            shape = "[1," + pb.split("tf_")[1].split("_")[2] + "," +  pb.split("tf_")[1].split("_")[3]+ ",3]"

            c_conv = ("python3 " + mo_file +
            " --input_model " + pb +
            " --output_dir " + save_folder +
            " --data_type FP16 " +
            " --input_shape " + shape +
            conv_cmd_str)
            xml_path = os.path.join(save_folder, pb.split(".pb")[0].split("/")[-1]+".xml")
        elif "cf_" in pb or "dk_" in pb:
            # Caffe or Darknet conversion
            # input_shape : batch, channels, width, height
            input_proto =  pb.split("/deploy.caffemodel")[0] + "/deploy.prototxt"
            if "cf_" in pb:
                shape = "[1,3," + pb.split("cf_")[1].split("_")[2] + "," +  pb.split("cf_")[1].split("_")[3]+ "]"
            elif "dk" in pb:
                shape = "[1,3," + pb.split("dk_")[1].split("_")[2] + "," + pb.split("dk_")[1].split("_")[3] + "]"

            if "SPnet" in pb:
                input_node = "demo"
            else:
                input_node = "data"

            c_conv = ("python3 " + mo_file +
            " --input_model " + pb +
            #" --input_proto " + input_proto +
            " --output_dir " + save_folder +
            " --data_type FP16 " +
            " --input_shape " + shape +
            " --input " + input_node + # input node sometimes called demo
            conv_cmd_str)

        if os.system(c_conv):
            print(c_conv)
            print("\nAn error has occured during conversion!\n")

        # set framework string and model name deploy.pb/forzen.pb
        framework = ""
        if "tf_" in pb:
            framework = "tf_"
            default_name = "frozen."
        elif "cf_" in pb:
            framework = "cf_"
            default_name = "deploy."
        elif "dk_" in pb:
            framework = "dk_"
            default_name = "deploy."

        # rename all three generated files
        extension_list = ["xml", "bin", "mapping"]
        for ex in extension_list:
            os.rename(os.path.join(save_folder, default_name + ex),
            os.path.join(save_folder, framework + pb.split(framework)[1].split("/")[0] + "." + ex))

        xml_path = os.path.join(save_folder, framework + pb.split(framework)[1].split("/")[0] + ".xml")

    model_name = xml_path.split(".xml")[0].split("/")[-1]

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    c_bench = ("python3 " + bench_app_file +
    " -m "  + xml_path +
    " -d MYRIAD " +
    " -b 1 " +
    " -api " + api +
    " -nireq 1 " +
    " -niter " + str(niter) +
    " --report_type average_counters" +
    " --report_folder " + report_dir)

    # start measurement in parallel to inference
    #daq_measurement(low_channel, high_channel, input_mode,ranges, samples_per_channel, rate, scan_options, flags, data)

    if uldaq_import and power_measurement == "True":
        x = threading.Thread(target=daq_measurement, args=(daq_device, low_channel, high_channel, input_mode,
                        ranges, samples_per_channel,
                        rate, scan_options, flags, data, data_dir, data_fname, index_pm, api))
        x.start()

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    new_avg_bench_path = os.path.join(report_dir, "_".join(("bacr", model_name.split(".pb")[0], str(index_pm), api,
                                                          ".csv")))
    new_stat_rep_path = os.path.join(report_dir, "_".join(("stat_rep", model_name.split(".pb")[0], str(index_pm), api,
                                                          ".csv")))
    # rename the default report file name
    if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_average_counters_report.csv"), new_avg_bench_path)
    if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_report.csv"), new_stat_rep_path)

    bench_over = True # this ends the power data gathering
    print("**********REPORTS GATHERED**********")

    return new_avg_bench_path, new_stat_rep_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCS2 power benchmark')
    parser.add_argument("-p", '--pb', default='yolov3.pb',
                        help='intermediade representation', required=False)
    parser.add_argument("-x", '--xml', default='yolov3.xml',
                        help='movidius representation', required=False)
    parser.add_argument("-sf", '--save_folder', default='save',
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-a", '--api', default='sync',
                        help='synchronous or asynchronous mode [sync, async]',
                        required=False)
    parser.add_argument("-n", '--niter', default=10,
                        help='number of iterations, useful in async mode', required=False)
    parser.add_argument("-pr", '--proto', default='caffe.proto',
                        help='Prototext for Xilinx Caffe', required=False)
    parser.add_argument("-rd", '--report_dir', default='reports',
                        help='Directory to save reports into', required=False)
    parser.add_argument("-pm", '--power_measurement', default='False',
                        help='parse "True" when conducting power measurements', required=False)
    args = parser.parse_args()

    index_run = 0

    if not args.pb and not args.xml:
        sys.exit("Please pass either a frozen pb or IR xml/bin model")

    if uldaq_import and args.power_measurement == "True":
        base = 500000
        daq_device, low_channel, high_channel, input_mode, meas_range, samples_per_channel,  rate, scan_options, flags, data, data_dir, data_fname = daq_setup(base,base*60)
    else:
        daq_device, low_channel, high_channel, input_mode, meas_range, \
        samples_per_channel, rate, scan_options, flags, data, data_dir, data_fname = None, None, None, None, None, None, \
                                                                                     None, None, None, None, None, None,

    run_bench(daq_device, low_channel, high_channel, input_mode, meas_range, samples_per_channel, rate, scan_options, flags,
              data, data_dir, data_fname,  args.power_measurement, index_run,
              xml=args.xml, pb=args.pb, save_folder=args.save_folder,
              report_dir=args.report_dir, niter=args.niter,
              api=args.api, proto=args.proto)
