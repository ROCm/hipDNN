# hipDNN Samples

## How to Build

1.  **Install hipDNN:** Follow the hipDNN build and install instructions in the [Building readme](../docs/Building.md).

2.  **Configure and Build:** Once hipDNN is installed, run the following commands from this `samples` directory:
    *   `mkdir build`
    *   `cd build`
    *   `cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ ..`
    *   `make -j`

3.  **Run Samples:** The `build` folder will now contain the sample executables.

## Available Samples

All samples are templated for mixed-precision execution with fp32, fp16, and bfp16 input/output data types, with fp32 intermediate accumulation. If desired, set `HIPDNN_LOG_LEVEL=info` environment variable to observe the detailed logs from the samples. The sample operations are:

*   **`bn_inference`** (`batchnorm/bn_inference.cpp`): Executes a single-node batch normalization inference graph.
    *   It normalizes each dimension of the input tensor `x` using the mean and inv_variance **population** statistics, and then scales the result by learned gamma and beta parameters. At a high-level, it computes the following linear transformation:
        ```
        y = gamma * (x - mean) * inv_variance + beta
        ```
        where `y` is propogated to the subsequent layer.
*   **`bn_training`** (`batchnorm/bn_training.cpp`): Executes the forward pass of a batch normalization training graph.
    *   For each input dimension, it computes the mean and inverse variance of the **mini-batch**. It then transforms the input and updates the running statistics:
        ```
        y = gamma * (x - mean) * inv_variance + beta
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_variance = (1 - momentum) * running_variance + momentum * batch_variance
        ```
    *   The graph outputs the normalized tensor `y`, along with the statistics (`saved_mean`, `saved_inv_variance`, `next_running_mean`, `next_running_variance`) required for the backward pass and inference.

*   **`bn_backward`** (`batchnorm/bn_backward.cpp`): Executes the backward pass of a batch normalization graph to compute gradients of the loss function.
    *   Given the upstream differentiable gradient `dy`, the downstream gradients are computed with the chain-rule:
        ```
        d_bias = sum(dy)
        x_hat = (x - saved_mean) * saved_inv_variance
        d_scale = sum(dy * x_hat)
        d_x = gamma * saved_inv_variance * (dy - (d_bias / m) - (x_hat * d_scale / m))
        ```
        where `m` is the size of the mini-batch, `d_x` is outputted and passed to the preceding layer, and `d_scale` and `d_bias` and a learning rate are used to update the learnable parameters `gamma` and `beta`.