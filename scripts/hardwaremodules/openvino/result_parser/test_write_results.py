import sys, os, json, argparse

import cv2
import logging as log
import numpy as np
import pandas as pd
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
        "--input",
        default="./input",
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
    images_hw = []
    for i in range(n):
        image = cv2.imread(args.input[i])
        image_height, image_width = image.shape[:-1]
        images_hw.append((image_height), (image_width))
        if image.shape[:1] != (h, w):
            log.warning(
                "Image {} is resized from {} to {}".format(
                    args.input[i], image.shape[:-1], (h, w)
                )
            )
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        images[i] = image

    print("Loading network")
    exec_net = ie.load_network(network=net, device_name="MYRIAD", num_requests=1)

    print("Starting inference")
    res = exec_net.infer(inputs={input_blob: images})
    # print(res)
    print("\nType of result object", type(res))

    res = res[out_blob]
    data = res[0][0]
    combined_data = []
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            image_id = np.int(proposal[0])
            image_height, image_width = images_hw[image_id]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(image_width * proposal[3])
            ymin = np.int(image_height * proposal[4])
            xmax = np.int(image_width * proposal[5])
            ymax = np.int(image_height * proposal[6])
            if proposal[2] > 0.5:
                combination_str = (
                    str(image_id)
                    + " "
                    + str(image_width)
                    + " "
                    + str(image_height)
                    + " "
                    + str(label)
                    + " "
                    + str(xmin)
                    + " "
                    + str(ymin)
                    + " "
                    + str(xmax)
                    + " "
                    + str(ymax)
                    + " "
                    + str(confidence)
                )
                combined_data.append([combination_str.strip()])

    dataframe = pd.DataFrame(
        combined_data,
        columns=[
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "score",
        ],
    )
    dataframe.to_csv("output" + ".csv", index=False)

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
