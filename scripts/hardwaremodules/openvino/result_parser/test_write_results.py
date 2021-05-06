import sys, os, json, argparse

import numpy as np
from openvino.inference_engine import IECore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCS2 settings test")
    parser.add_argument(
        "-m",
        "--model",
        default="./model.xml",
        help="model to test with",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-i",
        "--images",
        default="./images",
        help="images for the inference",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    model_name = args.model.split("/")[-1:][
        0
    ]  # extract model name from parsed model path

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

    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))

    # generate random input for network
    net_in = np.random.random(size=rand_size)

    print("Loading network")
    exec_net = ie.load_network(network=net, device_name="MYRIAD", num_requests=1)

    print("Starting inference")
    res = exec_net.infer(inputs={input_blob: net_in})
    # print(res)
    print("\nType of result object", type(res))

    res_new = {}
    for key, val in res.items():
        res_new = {key: val.tolist()}

    print("Saving results to file")
    with open("inf_res_" + model_name.split(".xml")[0] + ".json", "w") as f:
        json.dump(res_new, f)
        print("wrote result to file")

    print("Open saved dict again to check structure")
    with open("inf_res_" + model_name.split(".xml")[0] + ".json", "r") as f:
        res_load_back = json.load(f)
        print("read results")

    res_load_back_new = {}
    for key, val in res_load_back.items():
        # print(key + "\n")
        # print(np.asarray(val))
        res_load_back_new = {key: np.asarray(val)}  # same structure as original result
