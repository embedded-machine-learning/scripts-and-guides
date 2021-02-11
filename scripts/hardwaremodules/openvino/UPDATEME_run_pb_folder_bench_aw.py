# this script takes a folder with neural networks in the intermediate representation .pb
# and converts them to a Movidius NCS2 conform format with .xml and .bin
# and runs the openvino benchmark app on them
import pathlib

import tensorflow as tf
import os
import sys
import time

if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('pb_folder', 'models_to_test', 'folder with intermediate representations')
    flags.DEFINE_string('reports_folder', 'reports', 'folder to save the resulting files')
    flags.DEFINE_string('api', 'sync', 'synchronous or asynchronous mode [sync, async]')
    flags.DEFINE_string('niter', '4', 'number of iterations, useful in async mode')
    flags.DEFINE_string('device', 'CPU', 'Device [CPU, MYRIAD]')
    flags.DEFINE_string('ntests', '100', 'Number of tests to be executed')

    report_folder_long = os.path.abspath(FLAGS.reports_folder)

    models = None
    if not os.path.isdir(FLAGS.pb_folder):
        sys.exit("Please enter a valid directory!")
    else:
        models = sorted(os.listdir(FLAGS.pb_folder))
    if not os.path.isdir(FLAGS.reports_folder):
        os.mkdir(FLAGS.reports_folder)

    #remove all not xml-files from the list
    models_reduced = [i for i in models if 'xml' in i]

    for i, model_name in enumerate(models_reduced):
        print("++++++++++ {}: benching model {} on {} ++++++++++".format(i+1, model_name, FLAGS.device))
        beg = time.time()
        #c_bench_folder = ("python run_pb_bench.py " +
        #" --pb " + os.path.join(FLAGS.pb_folder, model_name) +
        #" --save_folder " + FLAGS.save_folder +
        #" --api " + FLAGS.api +
        #" --niter " + FLAGS.niter)

        model_path = os.path.abspath(pathlib.Path(FLAGS.pb_folder, model_name))

        for j in range(int(FLAGS.ntests)):
            print("***** Model: {}/{}, test {}/{} *****".format(i+1, len(models_reduced), j+1, FLAGS.ntests))
            c_bench_folder = ("python utils/run_pb_bench_win_NCS2.py " +
                              " -x " + "\"" +  model_path + "\"" +
                              " -rd " + "\"" +  FLAGS.reports_folder + "\"" +
                              " --device " + FLAGS.device +
                              " --api " + FLAGS.api +
                              " --niter " + FLAGS.niter)

            os.system(c_bench_folder)

    print("**********DONE**********")
