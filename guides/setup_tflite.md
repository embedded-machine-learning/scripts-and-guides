<h1>How to use tflite on a fresh raspberry os install.</h1>
The process is the same for 32 bit and 64 bit versions.<br>

<h2>BASICS</h2><br>

Install Raspberry OS on your device and boot to desktop. The site [Raspberry Pi OS](https://www.raspberrypi.org/software/) offers tools to install the OS quickly.<br>
To easily change your keyboard layout to German, click the icon in the left upper corner
and under Preferences > Keyboard and Mouse Layout choose German (or whatever other language).<br>
Connect to a wifi or ethernet of your choice using the GUI or the wpa supplicant. [The wifi setup process via GUI is described here](https://www.raspberrypi.org/documentation/configuration/wireless/desktop.md).<br>

To check your IP adress enter<br>
$ ifconfig

It is recommended to run<br>
$ sudo apt update<br>
$ sudo apt upgrade -y<br>
$ sudo apt autoremove<br>
To bring your system up to date. This may take a while, depending on the age your image and the speed of your device.<br>

<h3>(Optional)</h3>

if you want to work remotely, enable ssh by entering<br>
$ sudo raspbi-config<br>
and then under<br>
    3 Interface Options<br>
        P2 SSH<br>
            \<Yes\> \<No\><br>
choose the option Yes.<br>
SSH will be available after a restart<br>

<h3>TFLITE</h3>

There are a few methods to get tflite to run on a RPi. It can be built from source via cross compilation or directly on the Pi or a pre-compiled Python wheel can easily be installed via pip, if it is available for your platform.<br>
It is recommended to use a virtual environment (venv) of Python for better version control.<br>
To setup a venv, go to a directory of your choosing where the virtual environment will be setup and enter<br>
$ python3 -m venv venv_tflite<br>

This will use Python 3 as a base. To check which Python version you have enter
$ python3 --version<br>
Now a directory venv_tflite has been created in the current folder. The virtual environment can be activated by sourcing the activate file via<br>
$ source venv_tflite/bin/activate<br>
Now every package that will be installed via pip, will be contained in this directory
To check which packages are installed enter<br>
```pip list```

To update pip enter
$ pip install --upgrade pip

Now everything is ready to install the tflite interpreter
For example for Python3.7 and ARM 32 enter
$ pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
and after a few seconds it should be completed
Links to other tflite packages and how to run inference can be found in
https://www.tensorflow.org/lite/guide/python


To build TFlite from source, consider following this guide
https://qengineering.eu/install-tensorflow-2-lite-on-raspberry-pi-4.html
