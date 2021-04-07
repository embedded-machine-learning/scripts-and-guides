# for a sampling rate of 50000Hz on two channels (differential mode of gathering) the true rate is 50035.7 Hz


import argparse, os, sys, time, csv
import matplotlib.pyplot as plt
import numpy as np
import time


def visualize_data(data, layers_stamps, offset, name):
    time = np.arange(len(data)) / 500 # 500ks/second requires factor 500
    data_points = np.array(data)
    plt.plot(time, data_points, "-", linewidth=1, label="{}".format(name))
    plt.vlines(np.asarray(layers_stamps)+offset, min(data), max(data), linestyles='dashdot', colors="r", label="average counters")

    # add descriptive information to graph and show graph
    plt.title("Power Measurement")
    plt.xlabel("Time [ms]")
    plt.ylabel("Shunt Current [A]")
    #plt.legend(loc=0)

    plt.show()

def vis_cc(cc):
    cc_x = np.arange(len(cc))
    cc_y = np.array(cc)
    plt.plot(cc_x, cc_y, "-", linewidth=1, label="{Cross Correlation}")

    # add descriptive information to graph and show graph
    plt.title("Cross Correlation")
    #plt.legend(loc=0)

    plt.show()


def load_plain_data(filepath):
    with open(filepath, "rb") as f:
        data = f.readlines()
    return data


def load_numpy_data(filepath):
    with open(filepath, "rb") as f:
        data = np.load(f)
    return data


def threshold_data(data):
    return data[data > 0.02]


def truncate_data_with_rep_time(data, network_duration, extender, inf_num=10, sampling_rate=500000):
    return data[int(len(data) - (network_duration / 1000 * sampling_rate * inf_num * extender)) :] # return values from truncade_timestamp onwards


#correlate
#gradient

def prepare_data(data_file, report, delimiter, offset, cut, sampling_rate):
    # check validity of data file
    if not os.path.isfile(data_file):
        sys.exit("Could not find data under:", data_file)

    # extract name of network from data file name
    name = "".join(data_file.split(".")[:-1]).split("/")[-1]

    # load data
    beg_load_data = time.time()

    print("Using the numpy data format")
    data = load_numpy_data(data_file)

    print("loading {} rows of data took {:.2f} ms".format(len(data), (time.time() - beg_load_data) * 1000))

    # free the measurement data from all the bs
    if cut:
        data = threshold_data(data)

    data_len = len(data)

    # parse report data into layer_stamps if a report was passed
    layers_stamps = []
    energy = []

    if report:
        # check if passed report file exists
        if not os.path.isfile(report):
            sys.exit("Invalid report path: " + args.report + " given!")

        # get the data from file assuming an  openvino average counters benchmark report
        report_data = None
        with open(report, "r", newline='') as report_file:
            if delimiter == None:
                delimiter = ";"
            print("using ", delimiter, " as delimiter")
            report_data = list(csv.reader(report_file, delimiter=delimiter, quotechar='|'))
            # structure of data: ['layerName', 'execStatus', 'layerType', 'execType', 'realTime (ms)', 'cpuTime (ms)']

        # go over the data, cast it to float at append to a list
        layers_durations = []
        layer_names = []
        network_duration = 0
        for row in report_data:
            if row == []:
                break  # if line is empty, end loop
            #print(row, len(row))
            # check if the rows contains valid data in format xxx.xxx where the . signifies a float value
            if len(row) > 3 and "." in row[4]:
                if float(row[4]) > 0:
                    print(row[0], float(row[4]))
                    layer_names.append(row[0])
                    layers_durations.append(float(row[4]))
                if row[0] == "Total":
                    network_duration = float(row[4])
        print("\nnetwork duration: {0:5f} ms\n".format(network_duration))
        # prepare list for visualization
        time_stamp = 0
        layers_stamps = []
        for ld in layers_durations[0:-1]:
            time_stamp += ld
            layers_stamps.append(time_stamp)
            print(time_stamp)
        layers_stamps.insert(0, 0)

        offset_stamp = int(offset * sampling_rate / 1000)
        # convert time in [ms] to time stamps for each layer
        layer_power_stamps = []
        for lt in layers_stamps:
            if lt == 0:
                continue
            print("power stamps:", int((lt) * sampling_rate / 1000) + offset_stamp)
            layer_power_stamps.append(int((lt) * sampling_rate / 1000) + offset_stamp)

        if cut:
            extender = 10
            data = truncate_data_with_rep_time(data, network_duration, extender, inf_num=10, sampling_rate=sampling_rate)

        # go over all power stamps and sum the data values in between
        layer_power = []
        prev_lps = offset_stamp  # beginning power time stamp [0:50000]

        energy.append(("layer number", "layer name", "energy [mJ]"))
        for lnum, lps in enumerate(layer_power_stamps):
            # print("sum from:", prev_lps, "to:", lps)
            print("layer {} {}: {:.2f} [mJ]".format(lnum, layer_names[lnum], data[prev_lps: lps].sum() * 5))
            energy.append((lnum, layer_names[lnum], data[prev_lps: lps].sum() * 5))
            prev_lps = lps

    return data, layers_stamps, name, energy


def save_energy(name, energy):
    energy_dir = "energy"
    energy_file = name
    # create directory for energy data if it doesn't already exist
    if not os.path.isdir(energy_dir):
        os.mkdir(energy_dir)

    # save energy data to file with a meaningful name in the energy directory
    with open(os.path.join(energy_dir, energy_file + ".csv" ), "w") as f:
        writer = csv.writer(f, delimiter=";")
        for row in energy:
            print(row)
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCS2 settings test')
    parser.add_argument("-d", '--data_file',
                        default='data_dir/test_data',
                        help='file containing data', required=False)
    parser.add_argument("-r", '--report', type=str,
                        help='report containing average counters from benchmark app', required=False)
    parser.add_argument("-l", '--delimiter', type=str, default=";",
                        help='whatever the csv uses as delimiter', required=False)
    parser.add_argument("-o", '--offset',
                        default=0, type=float,
                        help='offset where to start drawing report results [ms]', required=False)
    parser.add_argument("-sr", '--sampling_rate',
                        default=500000, type=int,
                        help='sampling rate with which the data has been acquired', required=False)

    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.set_defaults(show=True)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=False)

    parser.add_argument('--cut', dest='cut', action='store_true')
    parser.add_argument('--no-cut', dest='cut', action='store_false')
    parser.set_defaults(cut=False)

    args = parser.parse_args()

    data, layers_stamps, name, energy = prepare_data(args.data_file, args.report, args.delimiter,
                                                     args.offset, args.cut, args.sampling_rate)

    # save the energy values per layer to a file
    for e in energy:
        print(e)
    if args.save:
        save_energy(name, energy)

    # visualize data and time stamps from the report
    if args.show:
        visualize_data(data, layers_stamps, args.offset, name)