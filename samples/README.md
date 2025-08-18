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

All samples are templated for mixed-precision execution with fp32, fp16, and bfp16 input/output data types, with fp32 intermediate accumulation. If desired, set `HIPDNN_LOG_LEVEL=info` to observe the detailed logs from the samples. The sample operations are:

*   **`bn_inference`** (`batchnorm/bn_inference.cpp`): Executes a single-node batch normalization inference graph on a 4D input tensor.
    *   It normalizes each dimension of the input tensor `x` of shape `(N, C, H, W)`, using pre-calculated population statistics. The result is then transformed by the learned parameters `scale` and `bias`, each with shape `(1, C, 1, 1)`. At a high-level, the following element-wise linear transformation is broadcast over the batch and spatial dimensions (`N, H, W`):
        ```
        y = scale * (x - mean) * inv_variance + bias
        ```
        where `y` would then be propogated as input to the subsequent layer.
*   **`bn_training`** (`batchnorm/bn_training.cpp`): Executes the forward pass of a batch normalization training graph on a 4D input tensor.
    *   For an input `x` of shape `(N, C, H, W)`, the mean and variance are calculated over the `N`, `H`, and `W` dimensions for each of the `C` channels or mini-batches, resulting in a `mean` and `inv_variance` of shape `(1, C, 1, 1)`. It then transforms the input and updates the running statistics:
        ```
        y = scale * (x - mean) * inv_variance + bias
        next_running_mean = (1 - momentum) * prev_running_mean + momentum * batch_mean
        next_running_variance = (1 - momentum) * prev_running_variance + momentum * batch_variance
        ```
    *   The graph outputs the normalized tensor `y`, along with the mini-batch statistics (`mean`, `inv_variance`) required for the backward pass, and the updated population statistics (`next_running_mean`, `next_running_variance`) required for inference.

*   **`bn_backward`** (`batchnorm/bn_backward.cpp`): Executes the backward pass of a batch normalization graph to compute gradients of the loss function.
    *   Given the upstream differentiable gradient `dy` of shape `(N, C, H, W)`, the downstream learnable gradients are computed with the chain-rule over the batch and spatial dimensions (`N, H, W`) with saved mini-batch statistics:
        ```
        dbias = sum(dy)
        x_hat = (x - mean) * inv_variance 
        dscale = sum(dy * x_hat)
        d_x = scale * inv_variance * (dy - (dbias / m) - (x_hat * dscale / m))
        ```
        where `m = N * H * W`, `d_x` would be passed to the preceding layer, and `d_scale` and `d_bias` can be used by an optimizer to update the learnable parameters `scale` and `bias`.