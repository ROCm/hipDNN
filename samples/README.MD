# hipDNN-samples

## How to build

1.  **Install hipDNN:** Follow the hipDNN build and install instructions in the [Building readme](../docs/Building.md).

2.  **Configure and Build:** Once hipDNN is installed, run the following commands from this `samples` directory:
    - `mkdir build`
    - `cd build`
    - `cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ ..`
    - `make -j`
    
3.  **Run Samples:** The `build` folder will now contain the sample executables.

## Available Samples

* `hipdnn_samples`: main.cpp - Basic demonstration of hipDNN
* `bn_infer`: (`bn_infer.cpp`) - Demonstrates a Batch Normalization inference graph.
* `bn_training`: (`bn_training.cpp`) - Demonstrates the forward pass of Batch Normalization training.
* `bn_backward`: (`bn_backward.cpp`) - Demonstrates the backward pass of Batch Normalization training.
