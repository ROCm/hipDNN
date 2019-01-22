A HIPDNN minimal deep learning training code sample using LeNet.

Prerequisites
=============

* C++11 capable compiler (Visual Studio 2013, GCC 4.9 etc.) (for chrono and random)
* For NV - CUDA (6.5 or newer): https://developer.nvidia.com/cuda-downloads
* HIPDNN
* MNIST dataset: http://yann.lecun.com/exdb/mnist/

Set the HIPDNN_PATH environment variable to where HIPDNN is installed.


Compilation
===========

To compile with CMake, run the following commands:
```bash
~: $ cd hipdnn-training/
~/hipdnn-training: $ mkdir build
~/hipdnn-training: $ cd build/
~/hipdnn-training/build: $ cmake ..
~/hipdnn-training/build: $ make
```

If compiling under linux, make sure to either set the ```HIPDNN_PATH``` environment variable to the path HIPDNN is installed.

Running
=======

Extract the MNIST training and test set files (*-ubyte) to a directory (default is the current path).

