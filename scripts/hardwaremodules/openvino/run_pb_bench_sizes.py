# this script takes a neural network in the intermediate representation .pd
# and converts it to a Movidius NCS2 conform format with .xml and .bin
# runs inference on the generated model

#import tensorflow as tf
from absl import flags, app
import os
import sys
FLAGS = flags.FLAGS
flags.DEFINE_string('pb', 'yolov3.pb', 'intermediade representation')
flags.DEFINE_string('xml', 'yolov3.xml', 'movidius representation')
flags.DEFINE_string('save_folder', './tmp/', 'folder to save the resulting files')
flags.DEFINE_string('api', 'sync', 'synchronous or asynchronous mode [sync, async]')
flags.DEFINE_string('niter', '100', 'number of iterations, useful in async mode')
flags.DEFINE_string('hw', 'MYRIAD', 'MYRIAD/CPU')
flags.DEFINE_string('size', '[1,224,224,3]', '[1,224,224,3]')

def main(argv):
    #flags = tf.app.flags

    mo_file = os.path.join("/", "opt", "intel", "openvino",
    "deployment_tools", "model_optimizer", "mo.py")
    bench_app_file = os.path.join("/","opt","intel", "openvino",
    "deployment_tools", "tools", "benchmark_tool", "benchmark_app.py")

    # check if necessary files exists
    if not os.path.isfile(mo_file) or not os.path.isfile(bench_app_file):
        sys.exit("Openvino not installed!")

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    print("\n**********Movidius FP16 conversion**********")
    xml_path = ""
    model_name = ""


    if not os.path.isfile(FLAGS.pb):
        if os.path.isfile(FLAGS.xml):
            xml_path = FLAGS.xml
            print("using already converted model! --> skipping conversion")
        else:
            sys.exit("Please enter a valid IR!")
    else:
        # yolov3/yolov3-tiny json file necessary for conversion
        conv_cmd_str = ""
        if "yolov3-tiny" in FLAGS.pb or "yolov3-tiny" in FLAGS.xml :
            conv_cmd_str = (" --tensorflow_use_custom_operations_config" +
            " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ yolo_v3_tiny.json")
        elif "yolov3" in FLAGS.pb or "yolov3-tiny" in FLAGS.xml :
            conv_cmd_str = (" --tensorflow_use_custom_operations_config" +
            " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ yolo_v3.json")

        if "tf_" in FLAGS.pb:
            # Tensorflow conversion
            # input_shape for tensorflow : batch, width, height, channels
            shape = "[1," + FLAGS.pb.split("tf_")[1].split("_")[2] + "," +  FLAGS.pb.split("tf_")[1].split("_")[3]+ ",3]"

            c_conv = ("python3 " + mo_file +
            " --input_model " + FLAGS.pb +
            " --output_dir " + FLAGS.save_folder +
            " --data_type FP16 " +
            " --input_shape " + shape +
            conv_cmd_str)
            xml_path = os.path.join(FLAGS.save_folder, FLAGS.pb.split(".pb")[0].split("/")[-1]+".xml")
        elif "cf_" in FLAGS.pb or "dk_" in FLAGS.pb:
            # Caffe or Darknet conversion
            # input_shape : batch, channels, width, height
            input_proto =  FLAGS.pb.split("/deploy.caffemodel")[0] + "/deploy.prototxt"
            if "cf_" in FLAGS.pb:
                shape = "[1,3," + FLAGS.pb.split("cf_")[1].split("_")[2] + "," +  FLAGS.pb.split("cf_")[1].split("_")[3]+ "]"
            elif "dk" in FLAGS.pb:
                shape = "[1,3," + FLAGS.pb.split("dk_")[1].split("_")[2] + "," + FLAGS.pb.split("dk_")[1].split("_")[3] + "]"

            if "SPnet" in FLAGS.pb:
                input_node = "demo"
            else:
                input_node = "data"

            c_conv = ("python3 " + mo_file +
            " --input_model " + FLAGS.pb +
            " --input_proto " + input_proto +
            " --output_dir " + FLAGS.save_folder +
            " --data_type FP16 " +
            " --input_shape " + shape +
            " --input " + input_node + # input node sometimes called demo
            conv_cmd_str)
        else:
            # Tensorflow conversion
            # input_shape for tensorflow : batch, width, height, channels
            #shape = "[1,513,1025,3]"

            c_conv = ("python3 " + mo_file +
            " --input_model " + FLAGS.pb +
            " --output_dir " + FLAGS.save_folder +
            " --data_type FP16 " +
            " --input_shape " + FLAGS.size +
            " --input x"+
            " --output Identity"+
            conv_cmd_str)
            xml_path = os.path.join(FLAGS.save_folder, FLAGS.pb.split(".pb")[0].split("/")[-1]+".xml")

        if os.system(c_conv):
            sys.exit("\nAn error has occured during conversion!\n")

        # set framework string and model name deploy.pb/forzen.pb
        framework = ""
        if "tf_" in FLAGS.pb:
            framework = "tf_"
            default_name = "frozen."
        elif "cf_" in FLAGS.pb:
            framework = "cf_"
            default_name = "deploy."
        elif "dk_" in FLAGS.pb:
            framework = "dk_"
            default_name = "deploy."
        else:
            framework = "tf_"
            default_name = "frozen_model."

        model_name = FLAGS.pb.split("/")[-1].split(".pb")[0]

        
        # rename all three generated files
        extension_list = ["xml", "bin", "mapping"]
        for ex in extension_list:
            os.rename(os.path.join(FLAGS.save_folder, model_name +"."+ ex),
            os.path.join(FLAGS.save_folder, framework + model_name +"."+ ex))

        xml_path = os.path.join(FLAGS.save_folder, framework + model_name + ".xml")

    model_name = xml_path.split(".xml")[0].split("/")[-1]
    # benchmark_app inference
    niter = FLAGS.niter
    api = FLAGS.api
    report_dir = "profiling_data"
    if api == "sync":
        report_dir += "_sync"
        niter_str = ""
    elif api == "async":
        report_dir += "_async_" + str(niter)
        niter_str = str(niter)

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    c_bench = ("python3 " + bench_app_file +
    " -m "  + xml_path +
    " -d " + FLAGS.hw +
    #" -b 1 " +
    " -api " + FLAGS.api +
    #" --exec_graph_path " + os.path.join(graph_dir, "graph") +
    " -niter " + str(niter) +
    " --report_type average_counters" +
    " --report_folder " + report_dir)

    if os.system(c_bench):
        sys.exit("An error has occured during benchmarking!")

    # rename the default report file name
    if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_average_counters_report.csv"),
        os.path.join(report_dir, "benchmark_average_counters_report_" +
        model_name.split(".pb")[0] + "_" + FLAGS.hw + "_" + str(api) + niter_str + ".csv"))
    if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_report.csv"),
        os.path.join(report_dir, "benchmark_report_" +
        model_name.split(".pb")[0] + "_" + FLAGS.hw + "_" + str(api) + niter_str +  ".csv"))

    print("**********DONE**********")

if __name__ == "__main__":
    app.run(main)

