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

    print("Loading network")
    exec_net = ie.load_network(network=net, device_name="MYRIAD", num_requests=1)

    combined_data = []
    _, _, net_h, net_w = net.input_info[input_blob].input_data.shape

    for filename in os.lisdir(args.input):
        original_image = cv2.imread(os.path.join(args.input, filename))
        image = original_image.copy()

        if image.shape[:-1] != (net_h, net_w):
            log.warning(
                f"Image {args.input} is resized from {image.shape[:-1]} to {(net_h, net_w)}"
            )
            image = cv2.resize(image, (net_w, net_h))

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        print("\nStarting inference for picture: " + filename)
        res = exec_net.infer(inputs={input_blob: image})

        # print(res)
        print("\nType of result object", type(res))

        output_image = original_image.copy()
        h, w, _ = output_image.shape

        if len(net.outputs) == 1:
            res = res[out_blob]
            # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
            # where N is the number of detected bounding boxes
            detections = res.reshape(-1, 7)
        else:
            detections = res["boxes"]
            labels = res["labels"]
            # Redefine scale coefficients
            w, h = w / net_w, h / net_h

        for i, detection in enumerate(detections):
            if len(net.outputs) == 1:
                _, class_id, confidence, xmin, ymin, xmax, ymax = detection
            else:
                class_id = labels[i]
                xmin, ymin, xmax, ymax, confidence = detection

            if confidence > 0.5:
                label = int(labels[class_id]) if args.labels else int(class_id)
                xmin = int(xmin * w)
                ymin = int(ymin * h)
                xmax = int(xmax * w)
                ymax = int(ymax * h)
                combination_str = (
                    str(filename)
                    + " "
                    + str(w)
                    + " "
                    + str(h)
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
