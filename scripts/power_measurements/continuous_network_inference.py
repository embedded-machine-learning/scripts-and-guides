import os
import sys
import argparse
import time

import numpy as np
from openvino.inference_engine import IECore



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCS2 settings test')
    parser.add_argument("-m", '--model', default='./model.xml',
                        help='model to test with', type=str, required=False)
    parser.add_argument('--one_per_second', dest='one_per_second', action='store_true')
    parser.add_argument('--no-one_per_second', dest='one_per_second', action='store_false')
    parser.set_defaults(one_per_second=True)
    args = parser.parse_args()

    model_name = args.model.split("/")[-1:][0]  # extract model name from parsed model path

    if not ".xml" in model_name:
        sys.exit("Invalid model xml given!")

    model_xml = args.model
    model_bin = args.model.split(".xml")[0] + ".bin"

    if not os.path.isfile(model_xml) or not os.path.isfile(model_bin):
        sys.exit("Could not find IR model for: " + model_xml)

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    in_blob = net.input_info[input_blob].input_data.shape
    net.input_info[input_blob].precision = "U8"
    net.batch_size = 1

    n, c, h, w = 0, 0, 0, 0
    if len(in_blob) == 4:
        n, c, h, w = in_blob
        rand_size = (n, c, h, w)
    if len(in_blob) == 3:
        c, h, w = in_blob
        rand_size = (c, h, w)

    # generate random input for network
    net_in = np.random.random(size=rand_size)

    print("Loading network")
    exec_net = ie.load_network(network=net, device_name="MYRIAD", num_requests=1)

    print("Starting inference")
    beg = 0
    try:
        while True:
            if args.one_per_second:
                if time.time() - beg > 1:
                    beg = time.time()
                    res = exec_net.infer(inputs={input_blob: net_in})
            else:
                res = exec_net.infer(inputs={input_blob: net_in})
            #print(res)
            #break

    except KeyboardInterrupt:
        print("\napplication ended with ctrl + c")
