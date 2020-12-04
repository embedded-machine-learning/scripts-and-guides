<h1>How to use the Tflite-Interpreter on a fresh Raspberry OS install.</h1>

The process is the same for fresh Raspberry OS 32 bit and 64 bit versions.

The non Raspberry OS specific parts can also be used for other Linux distributions, such as Ubuntu.

## <h2>BASICS</h2>

Install Raspberry OS on your device and boot to desktop. The site [Raspberry Pi OS](https://www.raspberrypi.org/software/) offers tools to install the OS quickly.

Connect to a wifi or ethernet of your choice using the GUI or the wpa supplicant. [The wifi setup process via GUI is described here](https://www.raspberrypi.org/documentation/configuration/wireless/desktop.md).

To check your IP adress enter
```
$ ifconfig
```
It is always recommended to run
```
sudo apt update
sudo apt upgrade -y
sudo apt autoremove
```
on a fresh install of any Linux distribution to bring your system up to date. This may take a while, depending on the age your image and the speed of your device.

## <h3>ssh on the Pi(Optional)</h3>

if you want to work remotely, enable ssh by entering
```
sudo raspbi-config
```
then under
1. Interface Options
2. P2 SSH
3. \<Yes\> \<No\>
4. choose the option Yes.

SSH will be available after a restart which can be invoked either with
```
sudo reboot
```
or through the GUI.

## <h3>Tflite interpreter setup</h3>

There are a few methods to get the Tflite-Interpreter to run on Linux. It can be built from source via cross compilation, built directly on the device, or a pre-compiled Python wheel can easily be installed via pip, if it is available for your platform.
It is recommended to use a Python **virtual environment** (venv) for better packet control.
Here, we will use Python 3 as base. To check which Python 3 version you are running and setup a venv, go to a directory of your choosing where the virtual environment will be setup and enter
```
python3 --version
python3 -m venv venv_tflite
```

Now a directory venv_tflite has been created in the current folder. The virtual environment can be activated by sourcing the activate file via
```
source venv_tflite/bin/activate
```
Now every package that will be installed via pip, will be contained in this directory.
To check which packages are installed enter
```
pip list
```

Pip in your venv will most likely be outdatet. To update pip enter
```
pip install --upgrade pip
```

Now everything is ready to install the tflite interpreter
For example for Python3.7 and ARM 32 enter
```
pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
```
and after a few seconds it should be completed.
Links to other Tflite packages and how to run inference can be found in
https://www.tensorflow.org/lite/guide/python


To build Tflite from source, consider following this guide
https://qengineering.eu/install-tensorflow-2-lite-on-raspberry-pi-4.html
