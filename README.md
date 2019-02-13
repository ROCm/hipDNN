# hipDNN

AMD's library (in beta stage) that supports a layer of abstraction around cuDNN and AMD's MIOpen.

## Prerequisites

1. [HIP](https://github.com/ROCm-Developer-Tools/HIP)
2. On AMD platforms, [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)
3. On Nvidia platforms, a functioning cuDNN installation.

## Build instructions
1. mkdir build
   cd build
   cmake ..
   make
2. The default installation path of the shared library is at /opr/rocm/hipdnn.

## General description

hipDNN defines a marshalling API between MIOpen-hipDNN, and cuDNN-hipDNN. Client programs only need to use the hipDNN API, and that will work on both nvidia and AMD platforms. On AMD(NVIDIA) platforms, the hipDNN datastructures are internally converted to appropriate MIOpen(cuDNN) datastructures, and the underlying library calls are made on behalf of the client. Results produced by those calls are mashalled back to hipDNN datastructures, so the calling program does not ever have to deal with the specific APIs.

hipDNN supports a very rich debugging model, which currently can be enabled at compile time by setting the DEBUG_CURRENT_CALL_STACK_LEVEL macro (currently supported on AMD platforms only). If enabled, the inputs of function calls, as well as the results (typically return by reference) are displayed to stdout.

In order to hipify a cuDNN program, it suffices to just:
+ Search and replace cudnn with hipdnn (typically for function calls and descriptors).
+ Search and replace CUDNN with HIPDNN (typically for enumerated types).
+ Include hipDNN.h, and link the DSO hipDNN.so

HIPDNN, and HIP overall, operate at compile time, i.e. you compile HIP programs for a particular platform. However hipDNN does not require the client programs to explicitly add platform specific header files and library paths to the makefiles, this is handled by hipDNN transparently.

## Comparing unit test results with Benchmark results

In order to verify the unit test results and evaluate the performance, checkout to the hipDNN branch comparison_unittest. We have used the results obtained by running the unit test in GeForce GTX 980 Ti (NVidia GPU) as benchmark results. These results are available in compressed folder NV_results.tar.gz. The python script compare_results.py in utils extracts these results inside hipDNN into a folder of the same name during execution and compares it with the results obtained during the execution of unit test every run. The final comparison results are saved as final_results.csv inside build folder in hipDNN.

## BUILD instructions

1. mkdir build && cd build
2. cmake ..
4. make -j4

## execution instructions

Inside build folder, run
        ./execute.sh
This shell script runs the unit test and compares the results with the benchmark results.
