<div align="center">
  <img src="../_img/eml_logo_and_text.png">
</div>

# Guide to compile Armnn from source with Pyarmnn as an option

This guides structure is based on other guides. None of the existing guides worked out of the box, so we found the need to write our own. Some commands were copied, many, especially in the later parts, werde modified to fit our needs.

These guides were used as inspiration:
* [Install ARMnn deep learning framework on a Raspberry Pi 4.](https://qengineering.eu/install-armnn-on-raspberry-pi-4.html)
* [Cross-compiling Arm NN for the Raspberry Pi and TensorFlow](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/cross-compiling-arm-nn-for-the-raspberry-pi-and-tensorflow)
* [Cross-compiling Arm NN for the Raspberry Pi and TensorFlow](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/cross-compiling-arm-nn-for-the-raspberry-pi-and-tensorflow/single-page)
* [ARM NN on RPi4](https://medium.com/@RouYunPan/arm-nn-on-rpi4-806ef8a10e61)

### Information

[This forum post](https://community.arm.com/developer/tools-software/graphics/f/discussions/12066/cross-compile-armnn-on-x86_64-for-arm64) has lots of useful information and references and [these tools](https://github.com/ARM-software/Tool-Solutions/tree/master/ml-tool-examples/build-armnn) can be used for a quick setup.

To install Flatbuffers on the host, you will need **sudo** permissions.

Throughout this guide, the compilation runs on 8 cores (threads), this is signified by the argument
```
-j $(nproc) # uses all available threads for compilation
```
in the compilation utilities. Change it to however many threads you want to run simultaneously. If run without the argument, the compilation utilities default to 1 thread, which usually takes quite a while to finish.

### On a new console

Should you for whatever reason start a new console session where you want to finish setting up the frameworks and go ahead with the compilation, please don't forget to set the environment variables for the shell as follows:

```
cd armnn-pi
export BASEDIR=$PWD
```
In case you came to the part where you compiled **boost** be sure to add

```
export PATH=$BASEDIR/boost.build/bin:$PATH
```

In case you came to the **protobuf** compilation, add

```
export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib/
```

(OPTIONAL) If you came as far as the **ONNX** setup, be sure to add

```
export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib:$LD_LIBRARY_PATH
```

[//]: # (this is a comment)

### (OPTIONAL) SWIG > 4

In case you get an error about swig not being installed or being too old during this guide, consult [swig setup](http://www.linuxfromscratch.org/blfs/view/svn/general/swig.html) and follow the instructions.

# Prepare directories and pull repositories
```
mkdir armnn-pi
cd armnn-pi
export BASEDIR=$PWD

# get the ARM libraries
git clone https://github.com/Arm-software/ComputeLibrary.git
git clone https://github.com/Arm-software/armnn

# and the dependencies
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
tar xf boost_1_64_0.tar.bz2
git clone -b v3.5.0 https://github.com/google/protobuf.git
wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
tar xf flatbuffers-1.12.0.tar.gz
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b
```
Whenever you start a new console where you want to work on this guide, please redo the steps
```
cd armnn-pi
export BASEDIR=$PWD
```
This will set the BASEDIR variable for the current console session, which is used extensively throughout this guide.

### Compile Compute Library

```
cd $BASEDIR/ComputeLibrary
```

if you're compiling for a 32 bit OS, use arch=armv7a
if you're compiling for a 64 bit OS, use arch=armv8a

```
scons -j $(nproc) \
extra_cxx_flags="-fPIC" \
Werror=0 debug=0 asserts=0 \
neon=1 opencl=0 os=linux arch=armv7a examples=1
```

### Compile Boost
```
cd $BASEDIR/boost_1_64_0/tools/build
./bootstrap.sh
./b2 install --prefix=$BASEDIR/boost.build

# incorporate the bin dir into PATH
export PATH=$BASEDIR/boost.build/bin:$PATH

# copy the user-config to project-config
cp $BASEDIR/boost_1_64_0/tools/build/example/user-config.jam $BASEDIR/boost_1_64_0/project-config.jam

cd $BASEDIR/boost_1_64_0
```

start an editor
```
nano project-config.jam
```
after

\"*Configure specific gcc version, giving alternative name to use.*

*using gcc : 3.2 : g++-3.2 ;*\"

 add the following line for a 32 Bit OS
```
using gcc : arm : arm-linux-gnueabihf-g++ ;
```
  or the following line for a 64 Bit OS
```
using gcc : arm : aarch64-linux-gnu-g++ ;
```
by copy pasting with \<Ctrl+C\>, \<Ctrl+Shift+V\> into the editor.
Then save with \<Ctrl+X\>, \<Y\>, \<ENTER\>*

```
b2 -j $(nproc) \
--build-dir=$BASEDIR/boost_1_64_0/build \
toolset=gcc-arm \
link=static \
cxxflags=-fPIC \
--with-filesystem \
--with-test \
--with-log \
--with-program_options install \
--prefix=$BASEDIR/boost
```

### Compile Protobuf for host system
```
cd $BASEDIR/protobuf
git submodule update --init --recursive
./autogen.sh
./configure --prefix=$BASEDIR/protobuf-host
make -j $(nproc)
make install
make clean
```

### Compile Protobuf for your RPi
```
cd $BASEDIR/protobuf
export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib/
```
for a 32 Bit OS use
```
./configure \
--prefix=$BASEDIR/protobuf-arm  \
--host=arm-linux  \
CC=arm-linux-gnueabihf-gcc \
CXX=arm-linux-gnueabihf-g++ \
--with-protoc=$BASEDIR/protobuf-host/bin/protoc
```
for a 64 Bit OS use
```
./configure --prefix=$BASEDIR/protobuf-arm --host=arm-linux CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ --with-protoc=$BASEDIR/protobuf-host/bin/protoc
```
```
make -j $(nproc)
make install
```

### Compile Tensorflow
 ```
cd $BASEDIR/tensorflow
../armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host
```

### Compile Flatbuffers for Host
```
cd $BASEDIR/flatbuffers-1.12.0
rm -f CMakeCache.txt
rm -rf build
mkdir build
cd build
CXXFLAGS="-fPIC" cmake .. \
-D FLATBUFFERS_BUILD_FLATC=1 \
-D CMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers
make -j $(nproc) all
sudo make install
```

### Compile Flatbuffers for RPi
```
cd $BASEDIR/flatbuffers-1.12.0
```
for a 32 Bit OS use
```
mkdir build-arm32
cd build-arm32
CXXFLAGS="-fPIC" cmake .. \
-D CMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
-D CMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
-D FLATBUFFERS_BUILD_FLATC=1 \
-D CMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-arm32 \
-D FLATBUFFERS_BUILD_TESTS=0
```

for a 64 Bit OS use
```
mkdir build-aarch64
cd build-aarch64
CXXFLAGS="-fPIC" cmake .. \
-D CMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
-D CMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
-D FLATBUFFERS_BUILD_FLATC=1 \
-D CMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-aarch64 \
-D FLATBUFFERS_BUILD_TESTS=0
```
```
make -j $(nproc) all
make install
```

### Flatbuffers for Tflite
```
cd $BASEDIR
mkdir tflite
cd tflite
cp $BASEDIR/tensorflow/tensorflow/lite/schema/schema.fbs .
$BASEDIR/flatbuffers-1.12.0/build/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs
```

## (OPTIONAL) Setup ONNX
If you plan on using models in the ONNX format or you are building Pyarmnn, you will need to setup ONNX first.
```
cd $BASEDIR/
git clone https://github.com/onnx/onnx.git
cd $BASEDIR/onnx/
git fetch https://github.com/onnx/onnx.git f612532843bd8e24efeab2815e45b436479cc9ab && git checkout FETCH_HEAD
export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib:$LD_LIBRARY_PATH
$BASEDIR/protobuf-host/bin/protoc \
onnx/onnx.proto --proto_path=. \
--proto_path=$BASEDIR/protobuf-host/include \
--cpp_out $BASEDIR/onnx
```

## (OPTIONAL) Setup Caffe for Pyarmnn
This part is only needed if you're planning on compiling the Python wrapper with Armnn where Caffe is required as a dependency.
```
# install Caffe dependencies
sudo apt install libhdf5-dev lmdb-utils liblmdb-dev \
libsnappy-dev libleveldb-dev python3-opencv libatlas-base-dev \
libgflags-dev libgoogle-glog-dev
```

### (OPTIONAL) Building OpenCV as a Caffe dependency
OpenCV is a powerful computer vision framework, which is required for Caffe.

```
cd $BASEDIR
wget https://github.com/opencv/opencv/archive/3.4.12.zip
unzip 3.4.12.zip
cd opencv-3.4.12/
mkdir build
cd build/
cmake .. -D  CMAKE_BUILD_TYPE=Release -D  CMAKE_INSTALL_PREFIX=/usr/local ..
make -j $(nproc)
sudo make install
```
To test your OpenCV installation, enter the following command
```
python3 -c "import cv2; print(cv2.__version__)"
```
it should output your opencv version.

### Caffe for Pyarmnn

Caffe uses the already prepared Boost and Protobuf
```
git clone https://github.com/BVLC/caffe.git
cd $BASEDIR/caffe
mkdir build
cd build/
```

### Cmake for Caffe
```
cmake .. \
-D CPU_ONLY=ON \
-D BOOST_ROOT=$BASEDIR/boost \
-D Protobuf_INCLUDE_DIR=$BASEDIR/protobuf-host/include \
-D Protobuf_PROTOC_EXECUTABLE=$BASEDIR/protobuf-host/bin/protoc \
-D BOOST_ROOT=$BASEDIR/boost \
-D PROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-host/lib/libprotobuf.so.15.0.0 \
-D BUILD_python=OFF \
-D OpenCV_DIR=$BASEDIR/opencv-3.4.12/cmake
#-D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#-D PYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
#-D PYTHON_EXECUTABLE=/usr/bin/python3
```

If you made a mistake or tried some flags which you want to reset, simply delete the CMakeCache by entering
```
rm CMakeCache.txt
```
in the build folder and rerun your cmake command with appropriate flags.

## Compiling Armnn
There are many options which can be enabled for the Armnn compilation. We tried to make a mostly complete version, so no missing dependencies would pop out later on during the deployment of Armnn.
To see the list of the available options enter:
```
cd $BASEDIR/armnn
cmake -LA
```
This should print a list of all the variables and their values. Via the cmake command, the arguments set the corresponding variables for the compilation.

Now it's time to build Armnn
```
mkdir build
cd build
```

### Armnn32 without Pyarmnn

```
cmake .. \
-D CMAKE_LINKER=/usr/bin/arm-linux-gnueabihf-ld \
-D CMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
-D CMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
-D CMAKE_C_COMPILER_FLAGS=-fPIC \
-D ARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-D ARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-D BOOST_ROOT=$BASEDIR/boost \
-D BUILD_TF_PARSER=1 \
-D TF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-D PROTOBUF_ROOT=$BASEDIR/protobuf-arm \
-D PROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-D PROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-D ARMCOMPUTENEON=1 \
-D BUILD_TESTS=1 \
-D ARMNNREF=1 \
-D BUILD_TF_LITE_PARSER=1 \
-D TF_LITE_GENERATED_PATH=$BASEDIR/tflite \
-D FLATBUFFERS_ROOT=$BASEDIR/flatbuffers-arm32 \
-D CMAKE_CXX_FLAGS=-mfpu=neon \
-D FLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
```

### Armnn32 with Pyarmnn
This time, the (OPTIONAL) software packets will be linked in the arguments as an addition to the flags from the compilation without Pyarmnn.

#### THIS COMPILATION WITH PYARMNN IS NOT YET SUCCESSFUL

```
cmake .. \
-D CMAKE_LINKER=/usr/bin/arm-linux-gnueabihf-ld \
-D CMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
-D CMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
-D CMAKE_C_COMPILER_FLAGS=-fPIC \
-D ARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-D ARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-D BOOST_ROOT=$BASEDIR/boost \
-D BUILD_TF_PARSER=1 \
-D TF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-D PROTOBUF_ROOT=$BASEDIR/protobuf-arm \
-D PROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-D PROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-D ARMCOMPUTENEON=1 \
-D BUILD_TESTS=1 \
-D ARMNNREF=1 \
-D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-D BUILD_PYTHON_SRC=ON \
-D BUILD_PYTHON_WHL=ON \
-D BUILD_TF_LITE_PARSER=1 \
-D TF_LITE_GENERATED_PATH=$BASEDIR/tflite \
-D FLATBUFFERS_ROOT=$BASEDIR/flatbuffers-arm32 \
-D CMAKE_CXX_FLAGS=-mfpu=neon \
-D FLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build \
-D BUILD_ONNX_PARSER=1 \
-D ONNX_GENERATED_SOURCES=$BASEDIR/onnx \
-D BUILD_CAFFE_PARSER:BOOL=ON
```
Might want to use following flags as well
```
BUILD_CAFFE_PARSER:BOOL=OFF
CAFFE_GENERATED_SOURCES:PATH=CAFFE_GENERATED_SOURCES-NOTFOUND

BUILD_ACCURACY_TOOL:BOOL=OFF
BUILD_ARMNN_QUANTIZER:BOOL=OFF
BUILD_ARMNN_SERIALIZER:BOOL=OFF
Boost_DIR:PATH=Boost_DIR-NOTFOUND

DYNAMIC_BACKEND_PATHS:BOOL=OFF
PROFILING_BACKEND_STREAMLINE:BOOL=OFF
SAMPLE_DYNAMIC_BACKEND:BOOL=OFF
SHARED_BOOST:BOOL=OFF
```

### Armnn64 without Pyarmnn

```
cmake .. \
-D CMAKE_LINKER=/usr/bin/aarch64-linux-gnu-ld \
-D CMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
-D CMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
-D CMAKE_C_COMPILER_FLAGS=-fPIC \
-D ARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-D ARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-D BOOST_ROOT=$BASEDIR/boost \
-D BUILD_TF_PARSER=1 \
-D TF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-D PROTOBUF_ROOT=$BASEDIR/protobuf-arm \
-D PROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-D PROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-D ARMCOMPUTENEON=1 \
-D BUILD_TESTS=1 \
-D ARMNNREF=1 .. \
-D BUILD_TF_LITE_PARSER=1 \
-D TF_LITE_GENERATED_PATH=$BASEDIR/tflite \
-D FLATBUFFERS_ROOT=$BASEDIR/flatbuffers-aarch64 \
-D FLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
```

### Armnn64 with Pyarmnn

#### THIS COMPILATION WITH PYARMNN IS NOT YET SUCCESSFUL

```
cmake ..
# top text
# insert command here
# bottom text
```

<h4>Compilation</h4>
After setting the options of cmake, all that's left is the compilation with

```
make -j $(nproc)
```

### Build problems
caffe is missing for pyarmnn
[Onnx build commands](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-onnx/single-page)

### PyArmnn issues and references
[Github issue with sample code](https://github.com/ARM-software/armnn/issues/468) might want to check out the results and use the code as future reference later.

## Deployment
To use Armnn you need to copy the compiled libraries onto your device. If you enabled ssh earlier in the guide, you can use a file transfer protocol of your choosing. Otherwise a fat32 partitioned usb drive is also an option.

You will need the **libprotobuf** files from protobuf-arm

*destination* is the folder where you want your libraries to be. For example it might be a folder in /home/pi or /usr/lib/

```
cp -r $BASEDIR/protobuf-arm/lib ./destination
cp -r $BASEDIR/armnn/build/lib* ./destination/lib/
```
Now you have to add the lib files to your library path on
```
cd path_to_lib/lib/
export LD_LIBRARY_PATH=$PWD
```

You can always check the path by typing
```
echo $LD_LIBRARY_PATH
```

Now you can copy and try out the test scripts on your Raspberry
```
cp -r $BASEDIR/armnn/build/tests/ ./destination
```
The program **ExecuteNetwork** takes a network and some other parameters as input, performs inference with 0s (if no input image/file given) and prints out the results as well as the inference time.

Firstly, **open a terminal session on your device** either via ssh or directly on the device. Then see if the program is executable, by entering

```
cd destination/
./ExecuteNetwork --help
```

This should print out all the options of the program. Try out the available options like CpuRef/CpuAcc or try printing the results of the intermediate layers to get a feel for the program.

For example a mobilenet_v2 network in the Tflite format can be run like this

```
./ExecuteNetwork \
--model-format tflite-binary \
--model-path ~/projects/models/mobilenet-v2-1.4-224/mobilenet_v2_1.4_224.tflite \
--input-name input \
--output-name MobilenetV2/Predictions/Reshape_1 \
--compute CpuAcc \
--iterations 10
```

Or if you have a frozen Tensorflow model, try
```
./ExecuteNetwork \
--model-format tensorflow-binary \
--model-path ~/projects/models/mobilenet-v2-1.4-224/mobilenet_v2_1.4_224_frozen.pb \
--input-name input \
--output-name MobilenetV2/Predictions/Reshape_1 \
--compute CpuAcc \
--iterations 10 \
-s 1,224,224,3
```

Should an error occur saying *libraries cannot be found*, make sure you copied all libraries from your host and check if the library path is set correctly. Should these be alright, check if the libraries were compiled for the right platform by first checking your OS with
```
uname -a
```
This will print your general OS information.

On a 32 bit OS, **arm7a** should be in the output

On a 64 bit OS, **aarch64** should be in the output

Now you can double check what platform the libraries were built for by using the  **file** command.

For example
```
file destination/lib/libprotobuf.so.15.0.0
```
You can check any other library which you copied to your device. There should be a platform (e.g. **arm7a**, **aarch64**) in the output.


## Issues
Should any issues arise during the completion of the guide or any errors noted, please let us know by filing an issue and help us keep up the quality.
