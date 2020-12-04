<h1>Guide to compile Armnn from source with Pyarmnn as an option </h1>

This guides structure is based on other guides. None of the existing guides worked out of the box, so we found the need to write our own. Some commands were copied, many, especially in the later parts, werde modified to fit our needs.

These guides were used as inspiration:
* [Install ARMnn deep learning framework on a Raspberry Pi 4.](https://qengineering.eu/install-armnn-on-raspberry-pi-4.html)
* [Cross-compiling Arm NN for the Raspberry Pi and TensorFlow](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/cross-compiling-arm-nn-for-the-raspberry-pi-and-tensorflow)
* [Cross-compiling Arm NN for the Raspberry Pi and TensorFlow](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/cross-compiling-arm-nn-for-the-raspberry-pi-and-tensorflow/single-page)
* [ARM NN on RPi4](https://medium.com/@RouYunPan/arm-nn-on-rpi4-806ef8a10e61)

# <h3>Information</h3>

[This forum post](https://community.arm.com/developer/tools-software/graphics/f/discussions/12066/cross-compile-armnn-on-x86_64-for-arm64) has lots of useful information and references and [these tools](https://github.com/ARM-software/Tool-Solutions/tree/master/ml-tool-examples/build-armnn) can be used for a quick setup.

To install Flatbuffers on the host, you will need **sudo** permissions.

Throughout this guide, the compilation runs on 8 cores (threads), this is signified by the argument
```
-j 8
```
in the compilation utilities. Change it to however many threads you want to run simultaneously. If run without the argument, the compilation utilities default to 1 thread, which usually takes quite a while to finish.

[//]: # (this is a comment)

# <h3>(OPTIONAL) SWIG > 4</h3>

In case you get an error about swig not being installed or being too old during this guide, consult [swig setup](http://www.linuxfromscratch.org/blfs/view/svn/general/swig.html) and follow the instructions.

# <h3>Prepare directories and pull repositories</h3>
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

# <h3>Compile Compute Library</h3>

```
cd $BASEDIR/ComputeLibrary
```

if you're compiling for a 32 bit OS, use arch=armv7a
if you're compiling for a 64 bit OS, use arch=armv8a

```
scons -j 8 \
extra_cxx_flags="-fPIC" \
Werror=0 debug=0 asserts=0 \
neon=1 opencl=0 os=linux arch=armv7a examples=1
```

# <h3>Compile Boost</h3>
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
b2 -j 8 \
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

# <h3>Compile Protobuf for host system</h3>
```
cd $BASEDIR/protobuf
git submodule update --init --recursive
./autogen.sh
./configure --prefix=$BASEDIR/protobuf-host
make -j 8
make install
make clean
```

# <h3>Compile Protobuf for your RPi</h3>
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
make -j 8
make install
```

# <h3>Compile Tensorflow</h3>
 ```
cd $BASEDIR/tensorflow
../armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host
```

# <h3>Compile Flatbuffers for Host</h3>
```
cd $BASEDIR/flatbuffers-1.12.0
rm -f CMakeCache.txt
rm -rf build
mkdir build
cd build
CXXFLAGS="-fPIC" cmake .. \
-DFLATBUFFERS_BUILD_FLATC=1 \
-DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers
make -j 8 all
sudo make install
```

# <h3>Compile Flatbuffers for RPi</h3>
```
cd $BASEDIR/flatbuffers-1.12.0
```
for a 32 Bit OS use
```
mkdir build-arm32
cd build-arm32
CXXFLAGS="-fPIC" cmake .. \
-DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
-DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
-DFLATBUFFERS_BUILD_FLATC=1 \
-DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-arm32 \
-DFLATBUFFERS_BUILD_TESTS=0
```

for a 64 Bit OS use
```
mkdir build-aarch64
cd build-aarch64
CXXFLAGS="-fPIC" cmake .. \
-DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
-DFLATBUFFERS_BUILD_FLATC=1 \
-DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-aarch64 \
-DFLATBUFFERS_BUILD_TESTS=0
```
```
make -j 8 all
make install
```

# <h3>Flatbuffers for Tflite</h3>
```
cd $BASEDIR
mkdir tflite
cd tflite
cp $BASEDIR/tensorflow/tensorflow/lite/schema/schema.fbs .
$BASEDIR/flatbuffers-1.12.0/build/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs
```

# <h2>(OPTIONAL) Setup ONNX for Pyarmnn</h2>

Pyarmnn requires ONNX as dependency.
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

# <h2>(OPTIONAL) Setup Caffe for Pyarmnn</h2>
This part is only needed if you're planning on compiling the Pyhton wrapper with Armnn.

Caffe is needed as a Pyarmnn dependency.
```
# install Caffe dependencies
sudo apt install libhdf5-dev lmdb-utils libsnappy-dev libleveldb-dev python3-opencv libatlas-base-dev
```

<h3>Building OpenCV as a Caffe dependence</h3>
OpenCV is a powerful computer vision framework, which is required for Caffe.

```
cd $BASEDIR
wget https://github.com/opencv/opencv/archive/3.4.12.zip
unzip 3.4.12.zip
cd opencv-3.4.12/
mkdir build
cd build/
cmake .. -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j 8
sudo make install
```
To test your OpenCV installation, enter the following command
```
python3 -c "import cv2; print(cv2.__version__)"
```
it should output your opencv version.

<h3>Caffe for Pyarmnn</h3>

Caffe uses the already prepared Boost and Protobuf
```
cd $BASEDIR/caffe
mkdir build
cd build/
cmake .. \
-DBOOST_ROOT=$BASEDIR/armnn-devenv/boost_arm64_install/ \
-DProtobuf_INCLUDE_DIR=$BASEDIR/armnn-devenv/google/x86_64_pb_install/include/ \
-DProtobuf_PROTOC_EXECUTABLE=$BASEDIR/armnn-devenv/google/x86_64_pb_install/bin/protoc \
-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DPYTHON_EXECUTABLE=/usr/bin/python3 \
-DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0 \
-DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0
```
# <h2>Compiling Armnn</h2>
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

# <h3>Armnn32 without Pyarmnn</h3>

```
cmake .. \
-DCMAKE_LINKER=/usr/bin/arm-linux-gnueabihf-ld \
-DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
-DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
-DCMAKE_C_COMPILER_FLAGS=-fPIC \
-DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-DBOOST_ROOT=$BASEDIR/boost \
-DBUILD_TF_PARSER=1 \
-DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-DPROTOBUF_ROOT=$BASEDIR/protobuf-arm \
-DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-DARMCOMPUTENEON=1 \
-DBUILD_TESTS=1 \
-DARMNNREF=1 \
-DBUILD_TF_LITE_PARSER=1 \
-DTF_LITE_GENERATED_PATH=$BASEDIR/tflite \
-DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-arm32 \
-DCMAKE_CXX_FLAGS=-mfpu=neon \
-DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
```

# <h3>Armnn32 with Pyarmnn</h3>
This time, the (OPTIONAL) software packets will be linked in the arguments as an addition to the flags from the compilation without Pyarmnn.

```
cmake .. \
-DCMAKE_LINKER=/usr/bin/arm-linux-gnueabihf-ld \
-DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
-DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
-DCMAKE_C_COMPILER_FLAGS=-fPIC \
-DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-DBOOST_ROOT=$BASEDIR/boost \
-DBUILD_TF_PARSER=1 \
-DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-DPROTOBUF_ROOT=$BASEDIR/protobuf-arm \
-DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-DARMCOMPUTENEON=1 \
-DBUILD_TESTS=1 \
-DARMNNREF=1 \
-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DBUILD_PYTHON_SRC=ON \
-DBUILD_PYTHON_WHL=ON \
-DBUILD_TF_LITE_PARSER=1 \
-DTF_LITE_GENERATED_PATH=$BASEDIR/tflite \
-DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-arm32 \
-DCMAKE_CXX_FLAGS=-mfpu=neon \
-DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build \
-DBUILD_ONNX_PARSER=1 \
-DONNX_GENERATED_SOURCES=$BASEDIR/onnx
```

# <h3>Armnn64 without Pyarmnn</h3>

```
cmake .. \
-DCMAKE_LINKER=/usr/bin/aarch64-linux-gnu-ld \
-DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
-DCMAKE_C_COMPILER_FLAGS=-fPIC \
-DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-DBOOST_ROOT=$BASEDIR/boost \
-DBUILD_TF_PARSER=1 \
-DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-DPROTOBUF_ROOT=$BASEDIR/protobuf-arm \
-DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0 \
-DARMCOMPUTENEON=1 \
-DBUILD_TESTS=1 \
-DARMNNREF=1 .. \
-DBUILD_TF_LITE_PARSER=1 \
-DTF_LITE_GENERATED_PATH=$BASEDIR/tflite \
-DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-aarch64 \
-DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
```

<h3>Armnn64 with Pyarmnn</h3>

```
cmake ..
# top text
# insert command here
# bottom text
```

<h3>Armnn64 with Pyarmnn ONLY for my work PC DELETE THIS WHEN PUBLISHING</h3>

```
cmake .. \
-DCMAKE_LINKER=/usr/bin/aarch64-linux-gnu-ld \
-DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
-DCMAKE_C_COMPILER_FLAGS=-fPIC \
-DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
-DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
-DBOOST_ROOT=$BASEDIR/armnn-devenv/boost_arm64_install/ \
-DBUILD_TF_PARSER=1 \
-DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf \
-DPROTOBUF_ROOT=$BASEDIR/armnn-devenv/google/x86_64_pb_install/ \
-DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0 \
-DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0 \
-DARMCOMPUTENEON=1 \
-DBUILD_TESTS=1 \
-DARMNNREF=1 \
-DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers \
-DFLATC_DIR=$BASEDIR/flatbuffers/ \
-DFLATBUFFERS_LIBRARY=$BASEDIR/flatbuffers/build/libflatbuffers.a \
-DBUILD_TF_LITE_PARSER=1 \
-DTF_LITE_GENERATED_PATH=$BASEDIR/tensorflow/tensorflow/lite/schema/ \
-DBUILD_ONNX_PARSER=1 \
-DONNX_GENERATED_SOURCES=$BASEDIR/onnx \
-DBUILD_CAFFE_PARSER=0 \
-DBUILD_PYTHON_SRC=ON \
-DBUILD_PYTHON_WHL=ON \
-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
```

```
-DCAFFE_GENERATED_SOURCES= # the location of the parent directory of caffe.pb.h and caffe.pb.cc
```
<h4>Compilation</h4>
After setting the options of cmake, all that's left is the compilation with

```
make -j 8
```

# <h3>Build problems</h3>
caffe is missing for pyarmnn
[Onnx build commands](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-onnx/single-page)

# <h2>Deployment</h2>
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
