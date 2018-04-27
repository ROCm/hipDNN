# hipDNN

AMD's library (in beta stage) that supports a layer of abstraction around cuDNN and AMD's MIOpen.

## Prerequisites

1. [HIP](https://github.com/ROCm-Developer-Tools/HIP)
2. On AMD platforms, [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)
3. On Nvidia platforms, a functioning cuDNN installation.

## Build instructions
1. make HIP_PATH=/your/path/to/hip/if/not/standard MIOPEN_PATH=/your/path/to/miopen/if/not/standard
2. The default installation path of the shared library is at /opr/rocm/hipDNN.  

## General description 

hipDNN defines a marshalling API between MIOpen-hipDNN, and cuDNN-hipDNN. Client programs only need to use the hipDNN API, and that will work on both nvidia and AMD platforms. On AMD(NVIDIA) platforms, the hipDNN datastructures are internally converted to appropriate MIOpen(cuDNN) datastructures, and the underlying library calls are made on behalf of the client. Results produced by those calls are mashalled back to hipDNN datastructures, so the calling program does not ever have to deal with the specific APIs.

hipDNN supports a very rich debugging model, which currently can be enabled at compile time by setting the DEBUG_CURRENT_CALL_STACK_LEVEL macro (currently supported on AMD platforms only). If enabled, the inputs of function calls, as well as the results (typically return by reference) are displayed to stdout.  

In order to hipify a cuDNN program, it suffices to just:
+ Search and replace cudnn with hipdnn (typically for function calls and descriptors).
+ Search and replace CUDNN with HIPDNN (typically for enumerated types).
+ Include hipDNN.h, and link the DSO hipDNN.so

HIPDNN, and HIP overall, operate at compile time, i.e. you compile HIP programs for a particular platform. However hipDNN does not require the client programs to explicitly add platform specific header files and library paths to the makefiles, this is handled by hipDNN transparently.



