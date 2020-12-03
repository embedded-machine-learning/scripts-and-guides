<h1>**Armnn compile**</h1>

based on [this guide](https://qengineering.eu/install-armnn-on-raspberry-pi-4.html)<br>


**Optional SWIG > 4**<br>
\# in case you get an error about swig not being installed or being too old,<br>
\# consult [swig setup](http://www.linuxfromscratch.org/blfs/view/svn/general/swig.html) and follow the instructions<br>

<h2>**Useful Information**</h2>

this link has lots of [useful information](https://community.arm.com/developer/tools-software/graphics/f/discussions/12066/cross-compile-armnn-on-x86_64-for-arm64)<br>
[these tools](https://github.com/ARM-software/Tool-Solutions/tree/master/ml-tool-examples/build-armnn) are quite useful for a quick setup<br><br>

<h3>Notes</h3>
To install Flatbuffers on the host, you will need sudo permissions<br>

**Prepare directories and pull repos**<br>
mkdir armnn-pi2<br>
cd armnn-pi2<br>
export BASEDIR=\`pwd\`<br>
\# get the ARM libraries<br>
git clone https://github.com/Arm-software/ComputeLibrary.git<br>
git clone https://github.com/Arm-software/armnn<br>
\# and the dependencies<br>
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2<br>
tar xf boost_1_64_0.tar.bz2<br>
git clone -b v3.5.0 https://github.com/google/protobuf.git<br>
wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz<br>
tar xf flatbuffers-1.12.0.tar.gz<br>
git clone https://github.com/tensorflow/tensorflow.git<br>
cd tensorflow<br>
git checkout 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b<br>
\# (OPTIONAL)<br>
cd $BASEDIR/<br>
git clone https://github.com/onnx/onnx.git<br>
cd $BASEDIR/onnx/<br>
git fetch https://github.com/onnx/onnx.git f612532843bd8e24efeab2815e45b436479cc9ab && git checkout FETCH_HEAD<br>

**Compile Compute Library**<br>
cd $BASEDIR/ComputeLibrary<br>
\# if you're compiling for a 32 bit OS, use arch=armv7a<br>
\# if you're compiling for a 64 bit OS, use arch=armv8a<br>
scons -j 8 extra_cxx_flags="-fPIC"
        Werror=0 debug=0 asserts=0 neon=1
        opencl=0 os=linux arch=armv7a examples=1<br>

**Compile Boost**<br>
\# run the scripts<br>
cd $BASEDIR/boost_1_64_0/tools/build<br>
./bootstrap.sh<br>
./b2 install --prefix=$BASEDIR/boost.build<br>
\# incorporate the bin dir into PATH<br>
export PATH=$BASEDIR/boost.build/bin:$PATH<br>

\# copy the user-config to project-config<br>
cp $BASEDIR/boost_1_64_0/tools/build/example/user-config.jam
     $BASEDIR/boost_1_64_0/project-config.jam<br>
\# change the directory<br>
cd $BASEDIR/boost_1_64_0<br>

*# start editor<br>
nano project-config.jam<br>
\# add the line for a 32 Bit OS<br>
**using gcc : arm : arm-linux-gnueabihf-g++ ;**<br>
\# or the line for a 64 Bit OS<br>
**using gcc : arm : aarch64-linux-gnu-g++ ;**<br>
after<br>
\# Configure specific gcc version, giving alternative name to use.<br>
\# using gcc : 3.2 : g++-3.2 ;<br>
in the project-config<br>
\# save with \<Ctrl+X\>, \<Y\>, \<ENTER\>*<br>

b2 -j 8
    --build-dir=$BASEDIR/boost_1_64_0/build
    toolset=gcc-arm
    link=static
    cxxflags=-fPIC
    --with-filesystem
    --with-test
    --with-log
    --with-program_options install
    --prefix=$BASEDIR/boost<br>

**Compile Protobuf for host system**<br>
cd $BASEDIR/protobuf<br>
git submodule update --init --recursive<br>
./autogen.sh<br>
./configure --prefix=$BASEDIR/protobuf-host<br>
make -j 8<br>
make install<br>
make clean<br>

**Compile Protobuf for Pi**<br>
cd $BASEDIR/protobuf<br>
export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib/<br>
\# for a 32 Bit OS use<br>
./configure --prefix=$BASEDIR/protobuf-arm --host=arm-linux CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ --with-protoc=$BASEDIR/protobuf-host/bin/protoc<br>

\# for a 64 Bit OS use<br>
./configure --prefix=$BASEDIR/protobuf-arm --host=arm-linux CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ --with-protoc=$BASEDIR/protobuf-host/bin/protoc<br>
make -j 8<br>
make install<br>

**Compile Tensorflow**<br>
\# tensorflow<br>
cd $BASEDIR/tensorflow<br>
../armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host<br>

**Setup ONNX (OPTIONAL)**<br>
\# ONNX is needed to compile pyarmnn<br>
cd $BASEDIR/onnx/<br>
export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib:$LD_LIBRARY_PATH<br>
$BASEDIR/protobuf-host/bin/protoc onnx/onnx.proto --proto_path=. --proto_path=$BASEDIR/protobuf-host/include --cpp_out $BASEDIR/onnx

**Compile Flatbuffers for Host**<br>
\# flatbuffers<br>
cd $BASEDIR/flatbuffers-1.12.0<br>
rm -f CMakeCache.txt<br>
rm -rf build<br>
mkdir build<br>
cd build<br>
CXXFLAGS="-fPIC" cmake .. -DFLATBUFFERS_BUILD_FLATC=1 -DCMAKE_INSTALL_PREFIX:PATH=$WORKING_DIR/flatbuffers<br>
make -j 8 all<br>
sudo make install<br>

\# **Compile Flatbuffers for Pi**<br>
cd $BASEDIR/flatbuffers-1.12.0<br>

\# for a 32 Bit OS use<br>
mkdir build-arm32<br>
cd build-arm32<br>
CXXFLAGS="-fPIC" cmake ..
-DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc
-DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++
-DFLATBUFFERS_BUILD_FLATC=1
-DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-arm32
-DFLATBUFFERS_BUILD_TESTS=0<br>

\# for a 64 Bit OS use<br>
mkdir build-aarch64<br>
cd build-aarch64<br>
CXXFLAGS="-fPIC" cmake ..
-DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc
-DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
-DFLATBUFFERS_BUILD_FLATC=1
-DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-aarch64
-DFLATBUFFERS_BUILD_TESTS=0<br>

make -j 8 all<br>
make install<br>

\# **flatbuffers for tflite**<br>
cd $BASEDIR<br>
mkdir tflite<br>
cd tflite<br>
cp $BASEDIR/tensorflow/tensorflow/lite/schema/schema.fbs .<br>
$BASEDIR/flatbuffers-1.12.0/build/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs<br>

\# **Caffe dependencies (OPTIONAL)**<br>
sudo apt install libhdf5-dev lmdb-utils libsnappy-dev libleveldb-dev python3-opencv libatlas-base-dev<br>

\# Building **opencv as a caffe** dependence<br>
cd $BASEDIR<br>
wget https://github.com/opencv/opencv/archive/3.4.12.zip<br>
unzip 3.4.12.zip<br>
cd opencv-3.4.12/<br>
mkdir build<br>
cd build/<br>
cmake .. -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..<br>
make -j 8<br>
sudo make install<br>
\# to test the installation, enter the following command <br>
python3 -c "import cv2; print(cv2.\_\_version\_\_)"<br>
\# it should output your opencv version<br>

\# building **Caffe for Pyarmnn**<br>
cd $BASEDIR/caffe<br>
mkdir build<br>
cd build/<br>
cmake ..
    -DBOOST_ROOT=$BASEDIR/armnn-devenv/boost_arm64_install/
    -DProtobuf_INCLUDE_DIR=$BASEDIR/armnn-devenv/google/x86_64_pb_install/include/
    -DProtobuf_PROTOC_EXECUTABLE=$BASEDIR/armnn-devenv/google/x86_64_pb_install/bin/protoc
    -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
    -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    -DPYTHON_EXECUTABLE=/usr/bin/python3.6
    -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0
    -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0
<br>

\# create a build directory<br>
cd $BASEDIR/armnn<br>
mkdir build<br>
cd build<br>

\# **armnn32 without pyarmnn** build command<br>
cmake -DCMAKE_LINKER=/usr/bin/arm-linux-gnueabihf-ld
        -DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc
        -DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++
        -DCMAKE_C_COMPILER_FLAGS=-fPIC
        -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary
        -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build
        -DBOOST_ROOT=$BASEDIR/boost
        -DBUILD_TF_PARSER=1
        -DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf
        -DPROTOBUF_ROOT=$BASEDIR/protobuf-arm
        -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0
        -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0
        -DARMCOMPUTENEON=1
        -DBUILD_TESTS=1
        -DARMNNREF=1 ..
        -DBUILD_TF_LITE_PARSER=1
        -DTF_LITE_GENERATED_PATH=$BASEDIR/tflite
        -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-arm32
        -DCMAKE_CXX_FLAGS=-mfpu=neon
        -DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
        <br>

\# **armnn32 with pyarmnn** build command<br>
cmake -DCMAKE_LINKER=/usr/bin/arm-linux-gnueabihf-ld
        -DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc
        -DCMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++
        -DCMAKE_C_COMPILER_FLAGS=-fPIC
        -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary
        -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build
        -DBOOST_ROOT=$BASEDIR/boost
        -DBUILD_TF_PARSER=1
        -DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf
        -DPROTOBUF_ROOT=$BASEDIR/protobuf-arm
        -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0
        -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0
        -DARMCOMPUTENEON=1
        -DBUILD_TESTS=1
        -DARMNNREF=1 ..
        -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
        -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
        -DBUILD_PYTHON_SRC=ON
        -DBUILD_PYTHON_WHL=ON
        -DBUILD_TF_LITE_PARSER=1
        -DTF_LITE_GENERATED_PATH=$BASEDIR/tflite
        -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-arm32
        -DCMAKE_CXX_FLAGS=-mfpu=neon
        -DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
        -DBUILD_ONNX_PARSER=1
        -DONNX_GENERATED_SOURCES=$BASEDIR/onnx
        <br>

\# **armnn64 without pyarmnn** build command<br>
cmake .. -DCMAKE_LINKER=/usr/bin/aarch64-linux-gnu-ld
        -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc
        -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
        -DCMAKE_C_COMPILER_FLAGS=-fPIC
        -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary
        -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build
        -DBOOST_ROOT=$BASEDIR/boost
        -DBUILD_TF_PARSER=1
        -DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf
        -DPROTOBUF_ROOT=$BASEDIR/protobuf-arm
        -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0
        -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/protobuf-arm/lib/libprotobuf.so.15.0.0
        -DARMCOMPUTENEON=1
        -DBUILD_TESTS=1
        -DARMNNREF=1 ..
        -DBUILD_TF_LITE_PARSER=1
        -DTF_LITE_GENERATED_PATH=$BASEDIR/tflite
        -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-aarch64
        -DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build
        <br>

\# **armnn64 with pyarmnn** build command for work PC<br>
  cmake
      -DCMAKE_LINKER=/usr/bin/aarch64-linux-gnu-ld
      -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc
      -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
      -DCMAKE_C_COMPILER_FLAGS=-fPIC
      -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary
      -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build
      -DBOOST_ROOT=$BASEDIR/armnn-devenv/boost_arm64_install/
      -DBUILD_TF_PARSER=1
      -DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf
      -DPROTOBUF_ROOT=$BASEDIR/armnn-devenv/google/x86_64_pb_install/
      -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0
      -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.0
      -DARMCOMPUTENEON=1
      -DBUILD_TESTS=1
      -DARMNNREF=1 ..
      -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers
      -DFLATC_DIR=$BASEDIR/flatbuffers/
      -DFLATBUFFERS_LIBRARY=$BASEDIR/flatbuffers/build/libflatbuffers.a
      -DBUILD_TF_LITE_PARSER=1
      -DTF_LITE_GENERATED_PATH=$BASEDIR/tensorflow/tensorflow/lite/schema/
      -DBUILD_ONNX_PARSER=1
      -DONNX_GENERATED_SOURCES=$BASEDIR/onnx
      -DBUILD_CAFFE_PARSER=0
      -DBUILD_PYTHON_SRC=ON
      -DBUILD_PYTHON_WHL=ON
      -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
      -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")<br>

  -DCAFFE_GENERATED_SOURCES= # the location of the parent directory of caffe.pb.h and caffe.pb.cc

  \# compile the cmake script<br>
  make -j 8<br>

\# caffe is missing for pyarmnn<br>
\# [Onnx build commands](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-onnx/single-page)<br>

\# TODO: how to describe what file to get where?
\# Which files to copy: to use ARMNN on your Raspberry you need to copy some libraries onto the device<br>

\# You will need the libprotobuf files from protobuf-arm<br>
\# destination is the folder where you want /lib to be<br>
cp -r $BASEDIR/protobuf-arm/lib ./destination<br>
cp -r $BASEDIR/armnn/build/lib* ./destination/lib/<br>
\# Now you have to add the lib files to your library path on your Raspberry<br>
cd path_to_lib/lib/<br>
export LD_LIBRARY_PATH=$PWD
\# You can always check if the path is correct by typing<br>
echo $LD_LIBRARY_PATH<br>

\# Now you can copy and try out the test scripts onto your Raspberry<br>
cp -r $BASEDIR/armnn/build/tests/ ./destination<br>
\# ExecuteNetwork takes a network and some parameters as input, performs inference with 0s at the input and prints out the results as well as the inference time.<br>
\# First see if the program is executable, by entering<br>
cd ./destination/<br>
./ExecuteNetwork --help
\# This should print out all the options of the program. Try playing around the options. For example a mobilenet_v2 network in the tflite format can be run like this<br>
./ExecuteNetwork --model-format tflite-binary --model-path ~/projects/models/mobilenet-v2-1.4-224/mobilenet_v2_1.4_224.tflite --input-name input --output-name MobilenetV2/Predictions/Reshape_1 --compute CpuAcc --iterations 10
\# Or if you have a frozen Tensorflow model, try something similar to this<br>
./ExecuteNetwork --model-format tensorflow-binary --model-path ~/projects/models/mobilenet-v2-1.4-224/mobilenet_v2_1.4_224_frozen.pb --input-name input --output-name MobilenetV2/Predictions/Reshape_1 --compute CpuAcc --iterations 10 -s 1,224,224,3<br>

\# Should an error occur saying libraries cannot be found, make sure you copied all libraries from your host and if the library path is set correctly. Should these be alright, check if the libraries were compiled for the right platform by first checking your OS with<br>
uname -a<br>
\# This will print your OS information.<br>
\# On a 32 bit OS, **arm7a** should be somewhere in the output<br>
\# On a 64 bit OS, **aarch64** should be somewhere in the output<br>

\# Now you can check what platform the libraries were built for by using the command *file*<br>
file ./destination/lib/libprotobuf.so.15.0.0
\# Or any other lib\* should have a platform (e.g. **arm7a**, **aarch64**) in the output.
