#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Wrapper call demonstrated:        ai_device.a_in_scan()

Purpose:                          Performs a continuous scan of the range
                                  of A/D input channels

Demonstration:                    Displays the analog input data for the
                                  range of user-specified channels using
                                  the first supported range and input mode

Steps:
1.  Call get_daq_device_inventory() to get the list of available DAQ devices
2.  Create a DaqDevice object
3.  Call daq_device.get_ai_device() to get the ai_device object for the AI
    subsystem
4.  Verify the ai_device object is valid
5.  Call ai_device.get_info() to get the ai_info object for the AI subsystem
6.  Verify the analog input subsystem has a hardware pacer
7.  Call daq_device.connect() to establish a UL connection to the DAQ device
8.  Call ai_device.a_in_scan() to start the scan of A/D input channels
9.  Call ai_device.get_scan_status() to check the status of the background
    operation
10. Display the data for each channel
11. Call ai_device.scan_stop() to stop the background operation
12. Call daq_device.disconnect() and daq_device.release() before exiting the
    process.
"""
from __future__ import print_function
from time import sleep, time
from os import system
from sys import stdout
import sys, os
import argparse
# from uldaq import create_float_buffer; help(create_float_buffer); create_float_buffer
from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag, ScanStatus,
                   ScanOption, create_float_buffer, InterfaceType, AiInputMode)
import matplotlib.pyplot as plt
import numpy as np


def main():
    # script example usage
    # python3 gather_and_save.py --sampling_rate 100000 --show --save --grow_fact 0.1
    parser = argparse.ArgumentParser(description='USB201 settings')
    parser.add_argument("-r", '--sampling_rate', default=500000, type=int,
                        help='number of samples to gather', required=False)
    parser.add_argument("-s", '--samples_per_channel', default=500000, type=int,
                        help='number of samples per channel', required=False)
    parser.add_argument("-sr", '--resistor', default=0.1, type=float,
                        help='value of the shunt resistor', required=False)

    # the most pythonic way to parse booleans with argparse is as follows
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parser.set_defaults(show=True)

    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=False)

    args = parser.parse_args()

    """Analog input scan example."""
    daq_device = None
    ai_device = None
    status = ScanStatus.IDLE

    # samples_per_channel (int): the number of A/D samples to collect from each channel in the scan.
    # rate (float): A/D sample rate in samples per channel per second.

    range_index = 0
    interface_type = InterfaceType.ANY
    low_channel = 0
    high_channel = 0
    samples_per_channel = args.samples_per_channel
    rate = args.sampling_rate
    scan_options = ScanOption.CONTINUOUS
    flags = AInScanFlag.DEFAULT
    data_already_saved = False

    try:
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

        data_dir = "data_dir"
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        # define name of a datafile and delete if it exists in the data directory
        data_fname = "test_data"
        if os.path.isfile(os.path.join(data_dir, data_fname)):
            os.remove(os.path.join(data_dir, data_fname))

        # Allocate a buffer to receive the data.
        data = create_float_buffer(channel_count, samples_per_channel)

        system('clear')

        # Start the acquisition.
        rate = ai_device.a_in_scan(low_channel, high_channel, input_mode,
                                   ranges[range_index], samples_per_channel,
                                   rate, scan_options, flags, data)

        if args.show:
            plt.ion()

            figure, ax = plt.subplots(figsize=(8, 6))
            line1, = ax.plot(np.arange(len(data))*(1000/rate), data)

            # add descriptive information to graph and show graph
            # plt.xticks(range(len(niter)),niter)
            plt.title("USB 201")
            plt.xlabel("Time [ms]")
            plt.ylabel("Shunt Current [A]")
            plt.legend(loc=0)

        ten_s_max, ten_s_min = 0, 0 # to reduce wobbling of the graph, set y-lims
        cnt = 0 # define a counter variable to know which read the loop is on
        try:
            while True:
                cnt += 1
                try:
                    # Get the status of the background operation
                    status, transfer_status = ai_device.get_scan_status()

                    reset_cursor()
                    print('Please enter CTRL + C to terminate the process\n')
                    print('Active DAQ device: ', descriptor.dev_string, ' (',
                          descriptor.unique_id, ')\n', sep='')

                    print('actual scan rate = ', '{:.1f}'.format(rate), 'Hz\n')

                    index = transfer_status.current_index

                    for i in range(channel_count):
                        clear_eol()
                        print('chan =',
                              i + low_channel, ': ',
                              '{:.6f} V'.format(data[index + i]))

                    if args.show:
                        # R_s = 0.1 ohm, I = U / R = data / 0.1 = data * 10
                        data_plt = np.asarray(data) / args.resistor

                        # scale the figure with data in the middle
                        grow_fact = 0.1
                        min_data, max_data = min(data_plt), max(data_plt)

                        if cnt % 5 == 0:
                            ten_s_max = -1000000
                            ten_s_min = 1000000
                        if max_data > ten_s_max:
                            ten_s_max = max_data
                        if min_data < ten_s_min:
                            ten_s_min = min_data

                        plt_offset = (ten_s_max - ten_s_min) / 2
                        if plt_offset != 0:
                            print(ten_s_min, ten_s_max, plt_offset)
                            plt.ylim(ten_s_min - plt_offset, ten_s_max + plt_offset)
                        #plt.ylim(1.4, 2.4)

                        line1.set_xdata(np.arange(len(data_plt))*(1000/rate))
                        line1.set_ydata(data_plt)
                        try:
                            figure.canvas.draw()
                            figure.canvas.flush_events()
                        except:
                            print("Exception in draw or flush_events methods. Exiting.")
                            break

                    if args.save and not data_already_saved and cnt == 1:
                        with open(os.path.join(data_dir, data_fname), "wb") as f:
                            np.save(f, np.asarray(data_plt))

                except (ValueError, NameError, SyntaxError):
                    break
        except KeyboardInterrupt:
            pass

    except RuntimeError as error:
        print('\n', error)

    finally:
        if daq_device:
            # Stop the acquisition if it is still running.
            if status == ScanStatus.RUNNING:
                ai_device.scan_stop()
            if daq_device.is_connected():
                daq_device.disconnect()
            daq_device.release()


def display_scan_options(bit_mask):
    """Create a displays string for all scan options."""
    options = []
    if bit_mask == ScanOption.DEFAULTIO:
        options.append(ScanOption.DEFAULTIO.name)
    for option in ScanOption:
        if option & bit_mask:
            options.append(option.name)
    return ', '.join(options)


def reset_cursor():
    """Reset the cursor in the terminal window."""
    stdout.write('\033[1;1H')


def clear_eol():
    """Clear all characters to the end of the line."""
    stdout.write('\x1b[2K')


if __name__ == '__main__':
    main()
