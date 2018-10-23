/*
 Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include "iostream"
#include <stdlib.h>
#include <time.h>
#include <hipdnn.h>
#include <nvcc_detail/hipdnn_cudnn.h>

hipdnnStatus_t cudnnTohipdnnStatus(cudnnStatus_t cStatus) {
    hipdnnStatus_t retVal;
    switch (cStatus) {
    case CUDNN_STATUS_SUCCESS:
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case CUDNN_STATUS_NOT_INITIALIZED:
        retVal = HIPDNN_STATUS_NOT_INITIALIZED;
        break;
    case CUDNN_STATUS_ALLOC_FAILED:
        retVal = HIPDNN_STATUS_ALLOC_FAILED;
        break;
    case CUDNN_STATUS_BAD_PARAM:
        retVal = HIPDNN_STATUS_BAD_PARAM;
        break;
    case CUDNN_STATUS_INTERNAL_ERROR:
        retVal = HIPDNN_STATUS_INTERNAL_ERROR;
        break;
    case CUDNN_STATUS_INVALID_VALUE:
        retVal = HIPDNN_STATUS_INVALID_VALUE;
        break;
    case CUDNN_STATUS_ARCH_MISMATCH:
        retVal = HIPDNN_STATUS_ARCH_MISMATCH;
        break;
    case CUDNN_STATUS_MAPPING_ERROR:
        retVal = HIPDNN_STATUS_MAPPING_ERROR;
        break;
    case CUDNN_STATUS_EXECUTION_FAILED:
        retVal = HIPDNN_STATUS_EXECUTION_FAILED;
        break;
    case CUDNN_STATUS_NOT_SUPPORTED:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case CUDNN_STATUS_LICENSE_ERROR:
        retVal = HIPDNN_STATUS_LICENSE_ERROR;
        break;
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
        retVal = HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING;
        break;
    default:
        retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    }

    return retVal;
}

cudnnStatus_t hipdnnTocudnnStatus(hipdnnStatus_t cStatus) {
    cudnnStatus_t retVal;
    switch (cStatus) {
    case HIPDNN_STATUS_SUCCESS:
        retVal = CUDNN_STATUS_SUCCESS;
        break;
    case HIPDNN_STATUS_NOT_INITIALIZED:
        retVal = CUDNN_STATUS_NOT_INITIALIZED;
        break;
    case HIPDNN_STATUS_ALLOC_FAILED:
        retVal = CUDNN_STATUS_ALLOC_FAILED;
        break;
    case HIPDNN_STATUS_BAD_PARAM:
        retVal = CUDNN_STATUS_BAD_PARAM;
        break;
    case HIPDNN_STATUS_INTERNAL_ERROR:
        retVal = CUDNN_STATUS_INTERNAL_ERROR;
        break;
    case HIPDNN_STATUS_INVALID_VALUE:
        retVal = CUDNN_STATUS_INVALID_VALUE;
        break;
    case HIPDNN_STATUS_ARCH_MISMATCH:
        retVal = CUDNN_STATUS_ARCH_MISMATCH;
        break;
    case HIPDNN_STATUS_MAPPING_ERROR:
        retVal = CUDNN_STATUS_MAPPING_ERROR;
        break;
    case HIPDNN_STATUS_EXECUTION_FAILED:
        retVal = CUDNN_STATUS_EXECUTION_FAILED;
        break;
    case HIPDNN_STATUS_NOT_SUPPORTED:
        retVal = CUDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_STATUS_LICENSE_ERROR:
        retVal = CUDNN_STATUS_LICENSE_ERROR;
        break;
    case HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
        retVal = CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING;
        break;
    default:
        retVal = CUDNN_STATUS_INTERNAL_ERROR;
    }

    return retVal;
}

hipdnnStatus_t hipTocudnnDataType(hipdnnDataType_t in, cudnnDataType_t *out) {
    switch (in) {
    case HIPDNN_DATA_FLOAT:
        *out = CUDNN_DATA_FLOAT;
        break;
    case HIPDNN_DATA_DOUBLE:
        *out = CUDNN_DATA_DOUBLE;
        break;
    case HIPDNN_DATA_HALF:
        *out = CUDNN_DATA_HALF;
        break;
    case HIPDNN_DATA_INT8:
        *out = CUDNN_DATA_INT8;
        break;
    case HIPDNN_DATA_INT32:
        *out = CUDNN_DATA_INT32;
        break;
    case HIPDNN_DATA_INT8x4:
        *out = CUDNN_DATA_INT8x4;
        break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t cudnnTohipDataType(cudnnDataType_t in, hipdnnDataType_t *out) {
    switch (in) {
    case CUDNN_DATA_FLOAT:
        *out = HIPDNN_DATA_FLOAT;
        break;
    case CUDNN_DATA_DOUBLE:
        *out = HIPDNN_DATA_DOUBLE;
        break;
    case CUDNN_DATA_HALF:
        *out = HIPDNN_DATA_HALF;
        break;
    case CUDNN_DATA_INT8:
        *out = HIPDNN_DATA_INT8;
        break;
    case CUDNN_DATA_INT32:
        *out = HIPDNN_DATA_INT32;
        break;
    case CUDNN_DATA_INT8x4:
        *out = HIPDNN_DATA_INT8x4;
        break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

//================================

hipdnnStatus_t hipTocudnnMathType(hipdnnMathType_t in, cudnnMathType_t *out) {
    switch (in) {
    case HIPDNN_DEFAULT_MATH:
        *out = CUDNN_DEFAULT_MATH;
        break;
    case HIPDNN_TENSOR_OP_MATH:
        *out = CUDNN_TENSOR_OP_MATH;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t cudnnTohipMathType(cudnnMathType_t in, hipdnnMathType_t *out) {
    switch (in) {
    case CUDNN_DEFAULT_MATH:
        *out = HIPDNN_DEFAULT_MATH;
        break;
    case CUDNN_TENSOR_OP_MATH:
        *out = HIPDNN_TENSOR_OP_MATH;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//================================

hipdnnStatus_t cudnnTohipOpTensorOp(cudnnOpTensorOp_t in,
                                    hipdnnOpTensorOp_t *out) {
    switch (in) {
    case CUDNN_OP_TENSOR_ADD:
        *out = HIPDNN_OP_TENSOR_ADD;
        break;
    case CUDNN_OP_TENSOR_MUL:
        *out = HIPDNN_OP_TENSOR_MUL;
        break;
    case CUDNN_OP_TENSOR_MIN:
        *out = HIPDNN_OP_TENSOR_MIN;
        break;
    case CUDNN_OP_TENSOR_MAX:
        *out = HIPDNN_OP_TENSOR_MAX;
        break;
    case CUDNN_OP_TENSOR_SQRT:
        *out = HIPDNN_OP_TENSOR_SQRT;
        break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipTocudnnOpTensorOp(hipdnnOpTensorOp_t in,
                                    cudnnOpTensorOp_t *out) {
    switch (in) {
    case HIPDNN_OP_TENSOR_ADD:
        *out = CUDNN_OP_TENSOR_ADD;
        break;
    case HIPDNN_OP_TENSOR_MUL:
        *out = CUDNN_OP_TENSOR_MUL;
        break;
    case HIPDNN_OP_TENSOR_MIN:
        *out = CUDNN_OP_TENSOR_MIN;
        break;
    case HIPDNN_OP_TENSOR_MAX:
        *out = CUDNN_OP_TENSOR_MAX;
        break;
    case HIPDNN_OP_TENSOR_SQRT:
        *out = CUDNN_OP_TENSOR_SQRT;
        break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

//===============================

hipdnnConvolutionMode_t cudnnTohipConvolutionMode(cudnnConvolutionMode_t in) {
    if (in == CUDNN_CONVOLUTION)
        return HIPDNN_CONVOLUTION;
    else if (in == CUDNN_CROSS_CORRELATION)
        return HIPDNN_CROSS_CORRELATION;

    return HIPDNN_CONVOLUTION;
}

cudnnConvolutionMode_t hipTocudnnConvolutionMode(hipdnnConvolutionMode_t in) {
    if (in == HIPDNN_CONVOLUTION)
        return CUDNN_CONVOLUTION;
    else if (in == HIPDNN_CROSS_CORRELATION)
        return CUDNN_CROSS_CORRELATION;

    return CUDNN_CONVOLUTION;
}

//=============================================================================

hipdnnStatus_t hipTocudnnPoolingMode(hipdnnPoolingMode_t in,
                                     cudnnPoolingMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_POOLING_MAX:
        *out = CUDNN_POOLING_MAX;
        break;
    case HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
        *out = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
    case HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
        *out = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        break;
    case HIPDNN_POOLING_MAX_DETERMINISTIC:
        *out = CUDNN_POOLING_MAX_DETERMINISTIC;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipPoolingMode(cudnnPoolingMode_t in,
                                     hipdnnPoolingMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_POOLING_MAX:
        *out = HIPDNN_POOLING_MAX;
        break;
    case CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
        *out = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
    case CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
        *out = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        break;
    case CUDNN_POOLING_MAX_DETERMINISTIC:
        *out = HIPDNN_POOLING_MAX_DETERMINISTIC;
        break;
    }
    return retVal;
}

//===================================

hipdnnStatus_t hipTocudnnLRNMode(hipdnnLRNMode_t in, cudnnLRNMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_LRN_WITHIN_CHANNEL:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_LRN_CROSS_CHANNEL:
        *out = CUDNN_LRN_CROSS_CHANNEL_DIM1;
        break;
    }
    return retVal;
}

//=============================================================================

cudnnBatchNormMode_t hipTocudnnBatchNormMode(hipdnnBatchNormMode_t in) {
    if (in == HIPDNN_BATCHNORM_PER_ACTIVATION)
        return CUDNN_BATCHNORM_PER_ACTIVATION;
    else if (in == HIPDNN_BATCHNORM_SPATIAL)
        return CUDNN_BATCHNORM_SPATIAL;
    else if (in == HIPDNN_BATCHNORM_SPATIAL_PERSISTENT)
        return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

    return CUDNN_BATCHNORM_PER_ACTIVATION;
}

//================================================

hipdnnStatus_t hipTocudnnActivationMode(hipdnnActivationMode_t in,
                                        cudnnActivationMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in) {
    case HIPDNN_ACTIVATION_SIGMOID:
        *out = CUDNN_ACTIVATION_SIGMOID;
        break;

    case HIPDNN_ACTIVATION_RELU:
        *out = CUDNN_ACTIVATION_RELU;
        break;

    case HIPDNN_ACTIVATION_TANH:
        *out = CUDNN_ACTIVATION_TANH;
        break;

    case HIPDNN_ACTIVATION_CLIPPED_RELU:
        *out = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;

    case HIPDNN_ACTIVATION_ELU:
    case HIPDNN_ACTIVATION_PATHTRU:
    case HIPDNN_ACTIVATION_SOFTRELU:
    case HIPDNN_ACTIVATION_ABS:
    case HIPDNN_ACTIVATION_POWER:
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipActivationMode(cudnnActivationMode_t in,
                                        hipdnnActivationMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in) {
    case CUDNN_ACTIVATION_SIGMOID:
        *out = HIPDNN_ACTIVATION_SIGMOID;
        break;

    case CUDNN_ACTIVATION_RELU:
        *out = HIPDNN_ACTIVATION_RELU;
        break;

    case CUDNN_ACTIVATION_TANH:
        *out = HIPDNN_ACTIVATION_TANH;
        break;

    case CUDNN_ACTIVATION_CLIPPED_RELU:
        *out = HIPDNN_ACTIVATION_CLIPPED_RELU;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//==========================================================
hipdnnStatus_t hipTocudnnConvolutionFwdAlgo(hipdnnConvolutionFwdAlgo_t in,
                                            cudnnConvolutionFwdAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        break;
    case HIPDNN_CONVOLUTION_FWD_ALGO_COUNT:
        *out = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipConvolutionFwdAlgo(cudnnConvolutionFwdAlgo_t in,
                                            hipdnnConvolutionFwdAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_GEMM;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_COUNT;
        break;
    }
    return retVal;
}

int ConvolutionFwdAlgoCount() { return (int)CUDNN_CONVOLUTION_FWD_ALGO_COUNT; }

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t GetConvolutionFwdAlgo(int i) {
    hipdnnConvolutionFwdAlgo_t retVal;
    cudnnConvolutionFwdAlgo_t cualgo;

    if (i < ConvolutionFwdAlgoCount()) {
        cualgo = (cudnnConvolutionFwdAlgo_t)i;
    } else {
        // for protection
        cualgo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    }
    cudnnTohipConvolutionFwdAlgo(
        cualgo, &retVal); // HGSOS does not compile, these functions need fwd
                          // declaration, or in hipdnn_cudnn header.

    return retVal;
}

//===========================================================

hipdnnStatus_t
hipTocudnnConvolutionBwdFilterAlgo(hipdnnConvolutionBwdFilterAlgo_t in,
                                   cudnnConvolutionBwdFilterAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

hipdnnStatus_t
cudnnTohipConvolutionBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t in,
                                   hipdnnConvolutionBwdFilterAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
        break;
    }
    return retVal;
}

int ConvolutionBwdFilterAlgoCount() {
    return (int)CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
}

// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t GetConvolutionBwdFilterAlgo(int i) {
    hipdnnConvolutionBwdFilterAlgo_t retVal;
    cudnnConvolutionBwdFilterAlgo_t cualgo;

    if (i < ConvolutionBwdFilterAlgoCount()) {
        cualgo = (cudnnConvolutionBwdFilterAlgo_t)i;
    } else {
        // for protection
        cualgo = (cudnnConvolutionBwdFilterAlgo_t)
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
    }
    cudnnTohipConvolutionBwdFilterAlgo(cualgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t
cudnnTohipConvolutionBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t in,
                                 hipdnnConvolutionBwdDataAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        break;
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        break;
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
        break;
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
        break;
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
        break;
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
        break;
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        break;
    }
    return retVal;
}

hipdnnStatus_t
hipTocudnnConvolutionBwdDataAlgo(hipdnnConvolutionBwdDataAlgo_t in,
                                 cudnnConvolutionBwdDataAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        break;
    }
    return retVal;
}

int ConvolutionBwdDataAlgoCount() {
    return (int)HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
}

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t GetConvolutionBwdDataAlgo(int i) {
    hipdnnConvolutionBwdDataAlgo_t retVal;
    cudnnConvolutionBwdDataAlgo_t cualgo;

    if (i < ConvolutionBwdDataAlgoCount()) {
        cualgo = (cudnnConvolutionBwdDataAlgo_t)i;
    } else {
        // for protection
        cualgo = (cudnnConvolutionBwdDataAlgo_t)
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
    }
    cudnnTohipConvolutionBwdDataAlgo(cualgo, &retVal);

    return retVal;
}
//=============================================================================

hipdnnStatus_t hipTocudnnConvolutionBwdDataPreference(
    hipdnnConvolutionBwdDataPreference_t in,
    cudnnConvolutionBwdDataPreference_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE:
        *out = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST:
        *out = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT:
        *out = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        break;
    }
    return retVal;
}

//==================================================

hipdnnStatus_t hipTocudnnSoftmaxAlgorithm(hipdnnSoftmaxAlgorithm_t in,
                                          cudnnSoftmaxAlgorithm_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_SOFTMAX_FAST:
        *out = CUDNN_SOFTMAX_FAST;
        break;
    case HIPDNN_SOFTMAX_ACCURATE:
        *out = CUDNN_SOFTMAX_ACCURATE;
        break;
    case HIPDNN_SOFTMAX_LOG:
        *out = CUDNN_SOFTMAX_LOG;
        break;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTocudnnSoftmaxMode(hipdnnSoftmaxMode_t in,
                                     cudnnSoftmaxMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_SOFTMAX_MODE_INSTANCE:
        *out = CUDNN_SOFTMAX_MODE_INSTANCE;
        break;
    case HIPDNN_SOFTMAX_MODE_CHANNEL:
        *out = CUDNN_SOFTMAX_MODE_CHANNEL;
        break;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTocudnnTensorFormat(hipdnnTensorFormat_t in,
                                      cudnnTensorFormat_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_TENSOR_NCHW:
        *out = CUDNN_TENSOR_NCHW;
        break;
    case HIPDNN_TENSOR_NHWC:
        *out = CUDNN_TENSOR_NHWC;
        break;
    case HIPDNN_TENSOR_NCHW_VECT_C:
        *out = CUDNN_TENSOR_NCHW_VECT_C;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipTensorFormat(cudnnTensorFormat_t in,
                                      hipdnnTensorFormat_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_TENSOR_NCHW:
        *out = HIPDNN_TENSOR_NCHW;
        break;
    case CUDNN_TENSOR_NHWC:
        *out = HIPDNN_TENSOR_NHWC;
        break;
    case CUDNN_TENSOR_NCHW_VECT_C:
        *out = HIPDNN_TENSOR_NCHW_VECT_C;
        break;
    }
    return retVal;
}

//===========================================================

hipdnnStatus_t hipTocudnnNanPropagation(hipdnnNanPropagation_t in,
                                        cudnnNanPropagation_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_NOT_PROPAGATE_NAN:
        *out = CUDNN_NOT_PROPAGATE_NAN;
        break;
    case HIPDNN_PROPAGATE_NAN:
        *out = CUDNN_PROPAGATE_NAN;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipNanPropagation(cudnnNanPropagation_t in,
                                        hipdnnNanPropagation_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_NOT_PROPAGATE_NAN:
        *out = HIPDNN_NOT_PROPAGATE_NAN;
        break;
    case CUDNN_PROPAGATE_NAN:
        *out = HIPDNN_PROPAGATE_NAN;
        break;
    }
    return retVal;
}

//=============================================================

hipdnnStatus_t
hipTocudnnConvolutionFwdPreference(hipdnnConvolutionFwdPreference_t in,
                                   cudnnConvolutionFwdPreference_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE:
        *out = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
        break;
    case HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST:
        *out = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
        break;
    case HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT:
        *out = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
        break;
    }
    return retVal;
}

hipdnnStatus_t hipTocudnnConvolutionBwdFilterPreference(
    hipdnnConvolutionBwdFilterPreference_t in,
    cudnnConvolutionBwdFilterPreference_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT:
        *out = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
        break;
    }
    return retVal;
}

// RNN Type Conversions

hipdnnStatus_t hipTocudnnRNNMode(hipdnnRNNMode_t in, cudnnRNNMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_RNN_RELU:
        *out = CUDNN_RNN_RELU;
        break;
    case HIPDNN_RNN_TANH:
        *out = CUDNN_RNN_TANH;
        break;
    case HIPDNN_LSTM:
        *out = CUDNN_LSTM;
        break;
    case HIPDNN_GRU:
        *out = CUDNN_GRU;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipRNNMode(cudnnRNNMode_t in, hipdnnRNNMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_RNN_RELU:
        *out = HIPDNN_RNN_RELU;
        break;
    case CUDNN_RNN_TANH:
        *out = HIPDNN_RNN_TANH;
        break;
    case CUDNN_LSTM:
        *out = HIPDNN_LSTM;
        break;
    case CUDNN_GRU:
        *out = HIPDNN_GRU;
        break;
    }
    return retVal;
}
//=============================================================

hipdnnStatus_t hipTocudnnDirectionMode(hipdnnDirectionMode_t in,
                                       cudnnDirectionMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_UNIDIRECTIONAL:
        *out = CUDNN_UNIDIRECTIONAL;
        break;
    case HIPDNN_BIDIRECTIONAL:
        *out = CUDNN_BIDIRECTIONAL;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipDirectionMode(cudnnDirectionMode_t in,
                                       hipdnnDirectionMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_UNIDIRECTIONAL:
        *out = HIPDNN_UNIDIRECTIONAL;
        break;
    case CUDNN_BIDIRECTIONAL:
        *out = HIPDNN_BIDIRECTIONAL;
        break;
    }
    return retVal;
}

//=============================================================

hipdnnStatus_t hipTocudnnRNNInputMode(hipdnnRNNInputMode_t in,
                                      cudnnRNNInputMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_LINEAR_INPUT:
        *out = CUDNN_LINEAR_INPUT;
        break;
    case HIPDNN_SKIP_INPUT:
        *out = CUDNN_SKIP_INPUT;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipdnnRNNInputMode(cudnnRNNInputMode_t in,
                                         hipdnnRNNInputMode_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_LINEAR_INPUT:
        *out = HIPDNN_LINEAR_INPUT;
        break;
    case CUDNN_SKIP_INPUT:
        *out = HIPDNN_SKIP_INPUT;
        break;
    }
    return retVal;
}

//=============================================================

hipdnnStatus_t hipTocudnnRNNAlgo(hipdnnRNNAlgo_t in, cudnnRNNAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_RNN_ALGO_STANDARD:
        *out = CUDNN_RNN_ALGO_STANDARD;
        break;
    case HIPDNN_RNN_ALGO_PERSIST_STATIC:
        *out = CUDNN_RNN_ALGO_PERSIST_STATIC;
        break;
    case HIPDNN_RNN_ALGO_PERSIST_DYNAMIC:
        *out = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipdnnRNNAlgo(cudnnRNNAlgo_t in, hipdnnRNNAlgo_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_RNN_ALGO_STANDARD:
        *out = HIPDNN_RNN_ALGO_STANDARD;
        break;
    case CUDNN_RNN_ALGO_PERSIST_STATIC:
        *out = HIPDNN_RNN_ALGO_PERSIST_STATIC;
        break;
    case CUDNN_RNN_ALGO_PERSIST_DYNAMIC:
        *out = HIPDNN_RNN_ALGO_PERSIST_DYNAMIC;
        break;
    }
    return retVal;
}

//=============================================================================

// CNTK 2.4

hipdnnStatus_t cudnnTohipReduceTensorOp(cudnnReduceTensorOp_t in,
                                        hipdnnReduceTensorOp_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_REDUCE_TENSOR_ADD:
        *out = HIPDNN_REDUCE_TENSOR_ADD;
        break;
    case CUDNN_REDUCE_TENSOR_MUL:
        *out = HIPDNN_REDUCE_TENSOR_MUL;
        break;
    case CUDNN_REDUCE_TENSOR_MIN:
        *out = HIPDNN_REDUCE_TENSOR_MIN;
        break;
    case CUDNN_REDUCE_TENSOR_MAX:
        *out = HIPDNN_REDUCE_TENSOR_MAX;
        break;
    case CUDNN_REDUCE_TENSOR_AMAX:
        *out = HIPDNN_REDUCE_TENSOR_AMAX;
        break;
    case CUDNN_REDUCE_TENSOR_AVG:
        *out = HIPDNN_REDUCE_TENSOR_AVG;
        break;
    case CUDNN_REDUCE_TENSOR_NORM1:
        *out = HIPDNN_REDUCE_TENSOR_NORM1;
        break;
    case CUDNN_REDUCE_TENSOR_NORM2:
        *out = HIPDNN_REDUCE_TENSOR_NORM2;
        break;
    case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
        *out = HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

hipdnnStatus_t hipTocudnnReduceTensorOp(hipdnnReduceTensorOp_t in,
                                        cudnnReduceTensorOp_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_REDUCE_TENSOR_ADD:
        *out = CUDNN_REDUCE_TENSOR_ADD;
        break;
    case HIPDNN_REDUCE_TENSOR_MUL:
        *out = CUDNN_REDUCE_TENSOR_MUL;
        break;
    case HIPDNN_REDUCE_TENSOR_MIN:
        *out = CUDNN_REDUCE_TENSOR_MIN;
        break;
    case HIPDNN_REDUCE_TENSOR_MAX:
        *out = CUDNN_REDUCE_TENSOR_MAX;
        break;
    case HIPDNN_REDUCE_TENSOR_AMAX:
        *out = CUDNN_REDUCE_TENSOR_AMAX;
        break;
    case HIPDNN_REDUCE_TENSOR_AVG:
        *out = CUDNN_REDUCE_TENSOR_AVG;
        break;
    case HIPDNN_REDUCE_TENSOR_NORM1:
        *out = CUDNN_REDUCE_TENSOR_NORM1;
        break;
    case HIPDNN_REDUCE_TENSOR_NORM2:
        *out = CUDNN_REDUCE_TENSOR_NORM2;
        break;
    case HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
        *out = CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t cudnnTohipReduceTensorIndices(cudnnReduceTensorIndices_t in,
                                             hipdnnReduceTensorIndices_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_REDUCE_TENSOR_NO_INDICES:
        *out = HIPDNN_REDUCE_TENSOR_NO_INDICES;
        break;
    case CUDNN_REDUCE_TENSOR_FLATTENED_INDICES:
        *out = HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }

    return retVal;
}

hipdnnStatus_t hipTocudnnReduceTensorIndices(hipdnnReduceTensorIndices_t in,
                                             cudnnReduceTensorIndices_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_REDUCE_TENSOR_NO_INDICES:
        *out = CUDNN_REDUCE_TENSOR_NO_INDICES;
        break;
    case HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES:
        *out = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }

    return retVal;
}

//=============================================================================

hipdnnStatus_t cudnnTohipIndicesType(cudnnIndicesType_t in,
                                     hipdnnIndicesType_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case CUDNN_32BIT_INDICES:
        *out = HIPDNN_32BIT_INDICES;
        break;
    case CUDNN_64BIT_INDICES:
        *out = HIPDNN_64BIT_INDICES;
        break;
    case CUDNN_16BIT_INDICES:
        *out = HIPDNN_16BIT_INDICES;
        break;
    case CUDNN_8BIT_INDICES:
        *out = HIPDNN_8BIT_INDICES;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

hipdnnStatus_t hipTocudnnIndicesType(hipdnnIndicesType_t in,
                                     cudnnIndicesType_t *out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_32BIT_INDICES:
        *out = CUDNN_32BIT_INDICES;
        break;
    case HIPDNN_64BIT_INDICES:
        *out = CUDNN_64BIT_INDICES;
        break;
    case HIPDNN_16BIT_INDICES:
        *out = CUDNN_16BIT_INDICES;
        break;
    case HIPDNN_8BIT_INDICES:
        *out = CUDNN_8BIT_INDICES;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle) {
    return cudnnTohipdnnStatus(cudnnCreate((cudnnHandle_t *)handle));
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle) {
    return cudnnTohipdnnStatus(cudnnDestroy((cudnnHandle_t)handle));
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId) {
    return cudnnTohipdnnStatus(
        cudnnSetStream((cudnnHandle_t)handle, (cudaStream_t)streamId));
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle,
                               hipdnnStream_t *streamId) {
    return cudnnTohipdnnStatus(
        cudnnGetStream((cudnnHandle_t)handle, (cudaStream_t *)streamId));
}

size_t hipdnnGetVersion() { return cudnnGetVersion(); }

hipdnnStatus_t
hipdnnCreateTensorDescriptor(hipdnnTensorDescriptor_t *tensorDesc) {
    return cudnnTohipdnnStatus(
        cudnnCreateTensorDescriptor((cudnnTensorDescriptor_t *)tensorDesc));
}

//===============================

hipdnnStatus_t hipdnnSetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
                                           hipdnnTensorFormat_t format,
                                           hipdnnDataType_t dataType, int n,
                                           int c, int h, int w) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnDataType_t cuDT;
    cudnnTensorFormat_t cuTF;

    retVal = hipTocudnnDataType(dataType, &cuDT);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = hipTocudnnTensorFormat(format, &cuTF);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnSetTensor4dDescriptor(
        (cudnnTensorDescriptor_t)tensorDesc, (cudnnTensorFormat_t)cuTF,
        (cudnnDataType_t)cuDT, n, c, h, w));
}

//=============================================================================

hipdnnStatus_t hipdnnGetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
                                           hipdnnDataType_t *dataType, int *n,
                                           int *c, int *h, int *w, int *nStride,
                                           int *cStride, int *hStride,
                                           int *wStride) {
    cudnnDataType_t cudT;
    hipdnnStatus_t retVal;

    retVal = cudnnTohipdnnStatus(cudnnGetTensor4dDescriptor(
        (cudnnTensorDescriptor_t)tensorDesc, (cudnnDataType_t *)&cudT, n, c, h,
        w, nStride, cStride, hStride, wStride));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipDataType(cudT, dataType);
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyTensorDescriptor(hipdnnTensorDescriptor_t tensorDesc) {
    return cudnnTohipdnnStatus((cudnnStatus_t)cudnnDestroyTensorDescriptor(
        (cudnnTensorDescriptor_t)tensorDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnAddTensor(hipdnnHandle_t handle, const void *alpha,
                               const hipdnnTensorDescriptor_t aDesc,
                               const void *A, const void *beta,
                               const hipdnnTensorDescriptor_t cDesc, void *C) {
    return cudnnTohipdnnStatus(cudnnAddTensor(
        (cudnnHandle_t)handle, alpha, (cudnnTensorDescriptor_t)aDesc, A, beta,
        (cudnnTensorDescriptor_t)cDesc, C));
}

//======================HGSOS======================!

hipdnnStatus_t hipdnnOpTensor(
    hipdnnHandle_t handle, const hipdnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const hipdnnTensorDescriptor_t aDesc, const void *A,
    const void *alpha2, const hipdnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const hipdnnTensorDescriptor_t cDesc, void *C) {

    return cudnnTohipdnnStatus(cudnnOpTensor(
        (cudnnHandle_t)handle, (cudnnOpTensorDescriptor_t)opTensorDesc, alpha1,
        (cudnnTensorDescriptor_t)aDesc, A, alpha2,
        (cudnnTensorDescriptor_t)bDesc, B, beta, (cudnnTensorDescriptor_t)cDesc,
        C));
}
//======

hipdnnStatus_t hipdnnSetTensor(hipdnnHandle_t handle,
                               const hipdnnTensorDescriptor_t yDesc, void *y,
                               const void *valuePtr) {
    return cudnnTohipdnnStatus(cudnnSetTensor(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)yDesc, y, valuePtr));
}

//==========

hipdnnStatus_t hipdnnScaleTensor(hipdnnHandle_t handle,
                                 const hipdnnTensorDescriptor_t yDesc, void *y,
                                 const void *alpha) {
    return cudnnTohipdnnStatus(cudnnScaleTensor(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)yDesc, y, alpha));
}
//=============================================================================

hipdnnStatus_t
hipdnnCreateFilterDescriptor(hipdnnFilterDescriptor_t *filterDesc) {
    return cudnnTohipdnnStatus(
        cudnnCreateFilterDescriptor((cudnnFilterDescriptor_t *)filterDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnCreateConvolutionDescriptor(hipdnnConvolutionDescriptor_t *convDesc) {
    return cudnnTohipdnnStatus(cudnnCreateConvolutionDescriptor(
        (cudnnConvolutionDescriptor_t *)convDesc));
}
//=====

hipdnnStatus_t
hipdnnSetConvolutionMathType(hipdnnConvolutionDescriptor_t convDesc,
                             hipdnnMathType_t mathType) {

    hipdnnStatus_t retVal;
    cudnnMathType_t cuMT;

    retVal = hipTocudnnMathType(mathType, &cuMT);
    if (retVal == HIPDNN_STATUS_SUCCESS)
        return cudnnTohipdnnStatus(cudnnSetConvolutionMathType(
            (cudnnConvolutionDescriptor_t)convDesc, (cudnnMathType_t)cuMT));
    return retVal;
}

hipdnnStatus_t hipdnnSetConvolution2dDescriptor(
    hipdnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v,
    int upscalex, int upscaley, hipdnnConvolutionMode_t mode,
    hipdnnDataType_t computeType) {

    hipdnnStatus_t retVal;
    cudnnDataType_t cuDT;

    retVal = hipTocudnnDataType(computeType, &cuDT);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnSetConvolution2dDescriptor(
        (cudnnConvolutionDescriptor_t)convDesc, pad_h, pad_w, u, v, upscalex,
        upscaley, hipTocudnnConvolutionMode(mode), (cudnnDataType_t)cuDT));
}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolution2dDescriptor(
    const hipdnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_y,
    int *u, int *v, int *upscalex, int *upscaley, hipdnnConvolutionMode_t *mode,
    hipdnnDataType_t *computeType) {
    hipdnnStatus_t retVal;
    cudnnConvolutionMode_t cuMode;
    cudnnDataType_t cutype;

    retVal = cudnnTohipdnnStatus(cudnnGetConvolution2dDescriptor(
        (cudnnConvolutionDescriptor_t)convDesc, pad_h, pad_y, u, v, upscalex,
        upscaley, &cuMode, &cutype));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    *mode = cudnnTohipConvolutionMode(cuMode);

    return cudnnTohipDataType(cutype, computeType);
}
//===========

hipdnnStatus_t hipdnnGetConvolution2dForwardOutputDim(
    const hipdnnConvolutionDescriptor_t convDesc, // HGSOSOS const
    const hipdnnTensorDescriptor_t inputTensorDesc,
    const hipdnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
    return cudnnTohipdnnStatus(cudnnGetConvolution2dForwardOutputDim(
        (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)inputTensorDesc,
        (cudnnFilterDescriptor_t)filterDesc, n, c, h, w));
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyConvolutionDescriptor(hipdnnConvolutionDescriptor_t convDesc) {
    return cudnnTohipdnnStatus(cudnnDestroyConvolutionDescriptor(
        (cudnnConvolutionDescriptor_t)convDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnFilterDescriptor_t wDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionFwdAlgoPerf_t *perfResults) {
    return cudnnTohipdnnStatus(cudnnFindConvolutionForwardAlgorithm(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc,
        (cudnnFilterDescriptor_t)wDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)yDesc, requestedAlgoCount, returnedAlgoCount,
        (cudnnConvolutionFwdAlgoPerf_t *)perfResults));
}

hipdnnStatus_t hipdnnGetConvolutionForwardAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnFilterDescriptor_t wDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc,
    hipdnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    hipdnnConvolutionFwdAlgo_t *algo) {
    cudnnConvolutionFwdAlgo_t cualgo;
    cudnnConvolutionFwdPreference_t cupref;
    hipdnnStatus_t retVal;

    retVal = hipTocudnnConvolutionFwdPreference(preference, &cupref);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = cudnnTohipdnnStatus(cudnnGetConvolutionForwardAlgorithm(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc,
        (cudnnFilterDescriptor_t)wDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)yDesc, cupref, memoryLimitInBytes, &cualgo));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipConvolutionFwdAlgo(cualgo, algo);
}

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithmEx(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {

    return cudnnTohipdnnStatus(cudnnFindConvolutionForwardAlgorithmEx(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc, x,
        (cudnnFilterDescriptor_t)wDesc, w,
        (cudnnConvolutionDescriptor_t)convDesc, (cudnnTensorDescriptor_t)yDesc,
        y, requestedAlgoCount, returnedAlgoCount,
        (cudnnConvolutionFwdAlgoPerf_t *)perfResults, workSpace,
        workSpaceSizeInBytes));
}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolutionForwardWorkspaceSize(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnFilterDescriptor_t wDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc, hipdnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
    cudnnConvolutionFwdAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionFwdAlgo(algo, &cualgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnGetConvolutionForwardWorkspaceSize(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc,
        (cudnnFilterDescriptor_t)wDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)yDesc, cualgo, sizeInBytes));
}

//=============================================================================

hipdnnStatus_t
hipdnnConvolutionForward(hipdnnHandle_t handle, const void *alpha,
                         const hipdnnTensorDescriptor_t xDesc, const void *x,
                         const hipdnnFilterDescriptor_t wDesc, const void *w,
                         const hipdnnConvolutionDescriptor_t convDesc,
                         hipdnnConvolutionFwdAlgo_t algo, void *workSpace,
                         size_t workSpaceSizeInBytes, const void *beta,
                         const hipdnnTensorDescriptor_t yDesc, void *y) {
    cudnnConvolutionFwdAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionFwdAlgo(algo, &cualgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnConvolutionForward(
        (cudnnHandle_t)handle, alpha, (cudnnTensorDescriptor_t)xDesc, x,
        (cudnnFilterDescriptor_t)wDesc, w,
        (cudnnConvolutionDescriptor_t)convDesc, cualgo, workSpace,
        workSpaceSizeInBytes, beta, (cudnnTensorDescriptor_t)yDesc, y));
}

//=============================================================================

hipdnnStatus_t
hipdnnConvolutionBackwardBias(hipdnnHandle_t handle, const void *alpha,
                              const hipdnnTensorDescriptor_t dyDesc,
                              const void *dy, const void *beta,
                              const hipdnnTensorDescriptor_t dbDesc, void *db) {
    return cudnnTohipdnnStatus(cudnnConvolutionBackwardBias(
        (cudnnHandle_t)handle, alpha, (cudnnTensorDescriptor_t)dyDesc, dy, beta,
        (cudnnTensorDescriptor_t)dbDesc, db));
}

//=============================================================================

hipdnnStatus_t hipdnnFindConvolutionBackwardFilterAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults) {

    return cudnnTohipdnnStatus(cudnnFindConvolutionBackwardFilterAlgorithm(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc,
        (cudnnTensorDescriptor_t)dyDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnFilterDescriptor_t)dwDesc, requestedAlgoCount, returnedAlgoCount,
        (cudnnConvolutionBwdFilterAlgoPerf_t *)perfResults));
}

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc,
    hipdnnConvolutionBwdFilterPreference_t preference,
    size_t memoryLimitInBytes, hipdnnConvolutionBwdFilterAlgo_t *algo) {
    cudnnConvolutionBwdFilterPreference_t cupreference;
    cudnnConvolutionBwdFilterAlgo_t cualgo;

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    retVal =
        hipTocudnnConvolutionBwdFilterPreference(preference, &cupreference);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = cudnnTohipdnnStatus(cudnnGetConvolutionBackwardFilterAlgorithm(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc,
        (cudnnTensorDescriptor_t)dyDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnFilterDescriptor_t)dwDesc, cupreference, memoryLimitInBytes,
        &cualgo));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipConvolutionBwdFilterAlgo(cualgo, algo);
}

hipdnnStatus_t hipdnnFindConvolutionBackwardFilterAlgorithmEx(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {

    return cudnnTohipdnnStatus(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc, x,
        (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnConvolutionDescriptor_t)convDesc, (cudnnFilterDescriptor_t)dwDesc,
        dw, requestedAlgoCount, returnedAlgoCount,
        (cudnnConvolutionBwdFilterAlgoPerf_t *)perfResults, workSpace,
        workSpaceSizeInBytes));
}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterWorkspaceSize(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc,
    hipdnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
    cudnnConvolutionBwdFilterAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionBwdFilterAlgo(algo, &cualgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        (cudnnHandle_t)handle, (cudnnTensorDescriptor_t)xDesc,
        (cudnnTensorDescriptor_t)dyDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnFilterDescriptor_t)dwDesc, cualgo, sizeInBytes));
}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardFilter(
    hipdnnHandle_t handle, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    hipdnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const hipdnnFilterDescriptor_t dwDesc, void *dw) {
    cudnnConvolutionBwdFilterAlgo_t cualgo;
    hipdnnStatus_t hstatus = hipTocudnnConvolutionBwdFilterAlgo(algo, &cualgo);
    if (hstatus != HIPDNN_STATUS_SUCCESS)
        return hstatus;
    return cudnnTohipdnnStatus(cudnnConvolutionBackwardFilter(
        (cudnnHandle_t)handle, alpha, (cudnnTensorDescriptor_t)xDesc, x,
        (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnConvolutionDescriptor_t)convDesc, cualgo, workSpace,
        workSpaceSizeInBytes, beta, (cudnnFilterDescriptor_t)dwDesc, dw));
}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolutionBackwardDataWorkspaceSize(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc, hipdnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
    cudnnConvolutionBwdDataAlgo_t cualgo;
    hipdnnStatus_t retVal;

    retVal = hipTocudnnConvolutionBwdDataAlgo(algo, &cualgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnGetConvolutionBackwardDataWorkspaceSize(
        (cudnnHandle_t)handle, (cudnnFilterDescriptor_t)wDesc,
        (cudnnTensorDescriptor_t)dyDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)dxDesc, cualgo, sizeInBytes));
}

//=============================================================================

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithm(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    return cudnnTohipdnnStatus(cudnnFindConvolutionBackwardDataAlgorithm(
        (cudnnHandle_t)handle, (cudnnFilterDescriptor_t)wDesc,
        (cudnnTensorDescriptor_t)dyDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)dxDesc, requestedAlgoCount, returnedAlgoCount,
        (cudnnConvolutionBwdDataAlgoPerf_t *)perfResults));
}

hipdnnStatus_t hipdnnGetConvolutionBackwardDataAlgorithm(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc,
    hipdnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes,
    hipdnnConvolutionBwdDataAlgo_t *algo) {
    cudnnConvolutionBwdDataPreference_t cupreference;
    cudnnConvolutionBwdDataAlgo_t cualgo;

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    retVal = hipTocudnnConvolutionBwdDataPreference(preference, &cupreference);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = cudnnTohipdnnStatus(cudnnGetConvolutionBackwardDataAlgorithm(
        (cudnnHandle_t)handle, (cudnnFilterDescriptor_t)wDesc,
        (cudnnTensorDescriptor_t)dyDesc, (cudnnConvolutionDescriptor_t)convDesc,
        (cudnnTensorDescriptor_t)dxDesc, cupreference, memoryLimitInBytes,
        &cualgo));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipConvolutionBwdDataAlgo(cualgo, algo);
}

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithmEx(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    hipdnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
    return cudnnTohipdnnStatus(cudnnFindConvolutionBackwardDataAlgorithmEx(
        (cudnnHandle_t)handle, (cudnnFilterDescriptor_t)wDesc, w,
        (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnConvolutionDescriptor_t)convDesc, (cudnnTensorDescriptor_t)dxDesc,
        dx, requestedAlgoCount, returnedAlgoCount,
        (cudnnConvolutionBwdDataAlgoPerf_t *)perfResults, workSpace,
        workSpaceSizeInBytes));
}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardData(
    hipdnnHandle_t handle, const void *alpha,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    hipdnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    cudnnConvolutionBwdDataAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionBwdDataAlgo(algo, &cualgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnConvolutionBackwardData(
        (cudnnHandle_t)handle, alpha, (cudnnFilterDescriptor_t)wDesc, w,
        (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnConvolutionDescriptor_t)convDesc, cualgo, workSpace,
        workSpaceSizeInBytes, beta, (cudnnTensorDescriptor_t)dxDesc, dx));
}

//=============================================================================

hipdnnStatus_t hipdnnSoftmaxForward(hipdnnHandle_t handle,
                                    hipdnnSoftmaxAlgorithm_t algo,
                                    hipdnnSoftmaxMode_t mode, const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x, const void *beta,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    void *y) {

    cudnnSoftmaxAlgorithm_t cuSMalgo;
    hipdnnStatus_t retVal = hipTocudnnSoftmaxAlgorithm(algo, &cuSMalgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    cudnnSoftmaxMode_t cuSMMode;
    retVal = hipTocudnnSoftmaxMode(mode, &cuSMMode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(
        cudnnSoftmaxForward((cudnnHandle_t)handle, cuSMalgo, cuSMMode, alpha,
                            (cudnnTensorDescriptor_t)xDesc, x, beta,
                            (cudnnTensorDescriptor_t)yDesc, y));
}

//=============================================================================

hipdnnStatus_t
hipdnnSoftmaxBackward(hipdnnHandle_t handle, hipdnnSoftmaxAlgorithm_t algo,
                      hipdnnSoftmaxMode_t mode, const void *alpha,
                      const hipdnnTensorDescriptor_t yDesc, const void *y,
                      const hipdnnTensorDescriptor_t dyDesc, const void *dy,
                      const void *beta, const hipdnnTensorDescriptor_t dxDesc,
                      void *dx) {

    cudnnSoftmaxAlgorithm_t cuSMalgo;
    hipdnnStatus_t retVal = hipTocudnnSoftmaxAlgorithm(algo, &cuSMalgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    cudnnSoftmaxMode_t cuSMMode;
    retVal = hipTocudnnSoftmaxMode(mode, &cuSMMode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnSoftmaxBackward(
        (cudnnHandle_t)handle, cuSMalgo, cuSMMode, alpha,
        (cudnnTensorDescriptor_t)yDesc, y, (cudnnTensorDescriptor_t)dyDesc, dy,
        beta, (cudnnTensorDescriptor_t)dxDesc, dx));
}

//=============================================================================

hipdnnStatus_t
hipdnnCreatePoolingDescriptor(hipdnnPoolingDescriptor_t *poolingDesc) {
    return cudnnTohipdnnStatus(
        cudnnCreatePoolingDescriptor((cudnnPoolingDescriptor_t *)poolingDesc));
}
//=============================================================================

hipdnnStatus_t hipdnnSetPooling2dDescriptor(
    hipdnnPoolingDescriptor_t poolingDesc, hipdnnPoolingMode_t mode,
    hipdnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride) {

    cudnnPoolingMode_t cuPMode;
    hipdnnStatus_t retVal = hipTocudnnPoolingMode(mode, &cuPMode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    cudnnNanPropagation_t cuNaN;
    retVal = hipTocudnnNanPropagation(maxpoolingNanOpt, &cuNaN);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnSetPooling2dDescriptor(
        (cudnnPoolingDescriptor_t)poolingDesc, cuPMode, cuNaN, windowHeight,
        windowWidth, verticalPadding, horizontalPadding, verticalStride,
        horizontalStride));
}

//=============================================================================

hipdnnStatus_t hipdnnGetPooling2dDescriptor(
    const hipdnnPoolingDescriptor_t poolingDesc, hipdnnPoolingMode_t *mode,
    hipdnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride) {
    hipdnnStatus_t retVal;
    cudnnPoolingMode_t cupmmode;
    cudnnNanPropagation_t cumaxpoolingNanOpt;

    retVal = cudnnTohipdnnStatus(cudnnGetPooling2dDescriptor(
        (cudnnPoolingDescriptor_t)poolingDesc, &cupmmode, &cumaxpoolingNanOpt,
        windowHeight, windowWidth, verticalPadding, horizontalPadding,
        verticalStride, horizontalStride));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = cudnnTohipPoolingMode(cupmmode, mode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipNanPropagation(cumaxpoolingNanOpt, maxpoolingNanOpt);
}

//=============================================================================

hipdnnStatus_t hipdnnGetPooling2dForwardOutputDim(
    const hipdnnPoolingDescriptor_t poolingDesc,
    const hipdnnTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h,
    int *w) {
    return cudnnTohipdnnStatus(cudnnGetPooling2dForwardOutputDim(
        (cudnnPoolingDescriptor_t)poolingDesc,
        (cudnnTensorDescriptor_t)inputTensorDesc, n, c, h, w));
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyPoolingDescriptor(hipdnnPoolingDescriptor_t poolingDesc) {
    return cudnnTohipdnnStatus(
        cudnnDestroyPoolingDescriptor((cudnnPoolingDescriptor_t)poolingDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingForward(
    hipdnnHandle_t handle, const hipdnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y) {
    return cudnnTohipdnnStatus(cudnnPoolingForward(
        (cudnnHandle_t)handle, (cudnnPoolingDescriptor_t)poolingDesc, alpha,
        (cudnnTensorDescriptor_t)xDesc, x, beta, (cudnnTensorDescriptor_t)yDesc,
        y));
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingBackward(
    hipdnnHandle_t handle, const hipdnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    return cudnnTohipdnnStatus(cudnnPoolingBackward(
        (cudnnHandle_t)handle, (cudnnPoolingDescriptor_t)poolingDesc, alpha,
        (cudnnTensorDescriptor_t)yDesc, y, (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnTensorDescriptor_t)xDesc, x, beta,
        (cudnnTensorDescriptor_t)dxDesc, dx));
}

//=============================================================================

hipdnnStatus_t
hipdnnCreateActivationDescriptor(hipdnnActivationDescriptor_t *activationDesc) {
    return cudnnTohipdnnStatus(cudnnCreateActivationDescriptor(
        (cudnnActivationDescriptor_t *)activationDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnSetActivationDescriptor(
    hipdnnActivationDescriptor_t activationDesc, // HGSOS const
    hipdnnActivationMode_t mode, hipdnnNanPropagation_t reluNanOpt,
    double reluCeilingOrAlpha, double activBeta, double activExp) {
    cudnnActivationMode_t cuAMode;
    hipdnnStatus_t retVal;
    cudnnNanPropagation_t cuNaN;

    retVal = hipTocudnnActivationMode(mode, &cuAMode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = hipTocudnnNanPropagation(reluNanOpt, &cuNaN);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnSetActivationDescriptor(
        (cudnnActivationDescriptor_t)activationDesc, // const
        cuAMode, cuNaN, reluCeilingOrAlpha));
}

//=============================================================================

hipdnnStatus_t hipdnnGetActivationDescriptor(
    const hipdnnActivationDescriptor_t activationDesc,
    hipdnnActivationMode_t *mode, hipdnnNanPropagation_t *reluNanOpt,
    double *reluCeilingOrAlpha, double *activBeta, double *activExp) {
    hipdnnStatus_t retVal;
    cudnnActivationMode_t cuactmode;
    cudnnNanPropagation_t cureluNanOpt;

    // not supported.
    *activBeta = 0.0;
    *activExp = 0.0;

    retVal = cudnnTohipdnnStatus(cudnnGetActivationDescriptor(
        (cudnnActivationDescriptor_t)activationDesc, &cuactmode, &cureluNanOpt,
        reluCeilingOrAlpha));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = cudnnTohipActivationMode(cuactmode, mode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipNanPropagation(cureluNanOpt, reluNanOpt);
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyActivationDescriptor(hipdnnActivationDescriptor_t activationDesc) {
    return cudnnTohipdnnStatus(cudnnDestroyActivationDescriptor(
        (cudnnActivationDescriptor_t)activationDesc));
}
//=================

hipdnnStatus_t hipdnnActivationForward(
    hipdnnHandle_t handle,
    hipdnnActivationDescriptor_t activationDesc, // HGSOS not const in cudnn.
    const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y) {
    return cudnnTohipdnnStatus(cudnnActivationForward(
        (cudnnHandle_t)handle, (cudnnActivationDescriptor_t)activationDesc,
        alpha, (cudnnTensorDescriptor_t)xDesc, x, beta,
        (cudnnTensorDescriptor_t)yDesc, y));
}
//======================

hipdnnStatus_t hipdnnActivationBackward(
    hipdnnHandle_t handle,
    hipdnnActivationDescriptor_t activationDesc, // HGSOS not const in cuda
    const void *alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    return cudnnTohipdnnStatus(cudnnActivationBackward(
        (cudnnHandle_t)handle, (cudnnActivationDescriptor_t)activationDesc,
        alpha, (cudnnTensorDescriptor_t)yDesc, y,
        (cudnnTensorDescriptor_t)dyDesc, dy, (cudnnTensorDescriptor_t)xDesc, x,
        beta, (cudnnTensorDescriptor_t)dxDesc, dx));
}
//=============================================================================

hipdnnStatus_t hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc) {
    return cudnnTohipdnnStatus(
        cudnnCreateLRNDescriptor((cudnnLRNDescriptor_t *)normDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnSetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
                                      hipdnnLRNMode_t mode, unsigned lrnN,
                                      double lrnAlpha, double lrnBeta,
                                      double lrnK) {

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnLRNMode_t cumode;

    retVal = hipTocudnnLRNMode(mode, &cumode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnSetLRNDescriptor(
        (cudnnLRNDescriptor_t)normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
}

//================

hipdnnStatus_t hipdnnGetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
                                      hipdnnLRNMode_t *mode, unsigned *lrnN,
                                      double *lrnAlpha, double *lrnBeta,
                                      double *lrnK) {

    *mode = HIPDNN_LRN_CROSS_CHANNEL;

    return cudnnTohipdnnStatus(cudnnGetLRNDescriptor(
        (cudnnLRNDescriptor_t)normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t lrnDesc) {
    return cudnnTohipdnnStatus(
        cudnnDestroyLRNDescriptor((cudnnLRNDescriptor_t)lrnDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelForward(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t yDesc, void *y) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnLRNMode_t cumode;

    retVal = hipTocudnnLRNMode(lrnMode, &cumode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnLRNCrossChannelForward(
        (cudnnHandle_t)handle, (cudnnLRNDescriptor_t)normDesc, cumode, alpha,
        (cudnnTensorDescriptor_t)xDesc, x, beta, (cudnnTensorDescriptor_t)yDesc,
        y));
}

hipdnnStatus_t hipdnnLRNCrossChannelForwardEx(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t yDesc, void *y, size_t workspacesize,
    void *workspace) {
    return hipdnnLRNCrossChannelForward(
        (cudnnHandle_t)handle, (cudnnLRNDescriptor_t)normDesc, lrnMode, alpha,
        (cudnnTensorDescriptor_t)xDesc, x, beta, (cudnnTensorDescriptor_t)yDesc,
        y);
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelBackward(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnLRNMode_t cumode;

    retVal = hipTocudnnLRNMode(lrnMode, &cumode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return cudnnTohipdnnStatus(cudnnLRNCrossChannelBackward(
        (cudnnHandle_t)handle, (cudnnLRNDescriptor_t)normDesc, cumode, alpha,
        (cudnnTensorDescriptor_t)yDesc, y, (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnTensorDescriptor_t)xDesc, x, beta,
        (cudnnTensorDescriptor_t)dxDesc, dx));
}

hipdnnStatus_t hipdnnLRNCrossChannelBackwardEx(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx, size_t workspacesize,
    void *workspace) {
    return hipdnnLRNCrossChannelBackward(
        (cudnnHandle_t)handle, (cudnnLRNDescriptor_t)normDesc, lrnMode, alpha,
        (cudnnTensorDescriptor_t)yDesc, y, (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnTensorDescriptor_t)xDesc, x, beta,
        (cudnnTensorDescriptor_t)dxDesc, dx);
}

//=============================================================================

hipdnnStatus_t
hipdnnDeriveBNTensorDescriptor(hipdnnTensorDescriptor_t derivedBnDesc,
                               const hipdnnTensorDescriptor_t xDesc,
                               hipdnnBatchNormMode_t mode) {
    return cudnnTohipdnnStatus(cudnnDeriveBNTensorDescriptor(
        (cudnnTensorDescriptor_t)derivedBnDesc, (cudnnTensorDescriptor_t)xDesc,
        hipTocudnnBatchNormMode(mode)));
}

//=============================================================================

hipdnnStatus_t hipdnnBatchNormalizationForwardTraining(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode, void *alpha, void *beta,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t yDesc, void *y,
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, void *bnScale,
    void *bnBias, double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance) {
    return cudnnTohipdnnStatus(cudnnBatchNormalizationForwardTraining(
        (cudnnHandle_t)handle, hipTocudnnBatchNormMode(mode), alpha, beta,
        (cudnnTensorDescriptor_t)xDesc, x, (cudnnTensorDescriptor_t)yDesc, y,
        (cudnnTensorDescriptor_t)bnScaleBiasMeanVarDesc, bnScale, bnBias,
        exponentialAverageFactor, resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance));
}
//=============================================================================

hipdnnStatus_t hipdnnnBatchNormalizationForwardInference(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode, void *alpha, void *beta,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t yDesc, void *y,
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    return cudnnTohipdnnStatus(cudnnBatchNormalizationForwardInference(
        (cudnnHandle_t)handle, hipTocudnnBatchNormMode(mode), alpha, beta,
        (cudnnTensorDescriptor_t)xDesc, x, (cudnnTensorDescriptor_t)yDesc, y,
        (cudnnTensorDescriptor_t)bnScaleBiasMeanVarDesc, bnScale, bnBias,
        estimatedMean, estimatedVariance, epsilon));
}
//=============================================================================

hipdnnStatus_t hipdnnBatchNormalizationBackward(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t dxDesc, void *dx,
    const hipdnnTensorDescriptor_t bnScaleBiasDiffDesc, const void *bnScale,
    void *resultBnScaleDiff, void *resultBnBiasDiff, double epsilon,
    const void *savedMean, const void *savedInvVariance) {
    return cudnnTohipdnnStatus(cudnnBatchNormalizationBackward(
        (cudnnHandle_t)handle, hipTocudnnBatchNormMode(mode), alphaDataDiff,
        betaDataDiff, alphaParamDiff, betaParamDiff,
        (cudnnTensorDescriptor_t)xDesc, x, (cudnnTensorDescriptor_t)dyDesc, dy,
        (cudnnTensorDescriptor_t)dxDesc, dx,
        (cudnnTensorDescriptor_t)bnScaleBiasDiffDesc, bnScale,
        resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean,
        savedInvVariance));
}

hipdnnStatus_t hipdnnSetTensorNdDescriptor(hipdnnTensorDescriptor_t tensorDesc,
                                           hipdnnDataType_t dataType,
                                           int nbDims, const int dimA[],
                                           const int strideA[]) {

    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;
    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnSetTensorNdDescriptor(
        (cudnnTensorDescriptor_t)tensorDesc, cuDT, nbDims, dimA, strideA));
}

hipdnnStatus_t
hipdnnGetTensorNdDescriptor(const hipdnnTensorDescriptor_t tensorDesc,
                            int nbDimsRequested, hipdnnDataType_t *dataType,
                            int *nbDims, int dimA[], int strideA[]) {
    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;
    retval = cudnnTohipdnnStatus(cudnnGetTensorNdDescriptor(
        (cudnnTensorDescriptor_t)tensorDesc, nbDimsRequested, &cuDT, nbDims,
        dimA, strideA));
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;
    return cudnnTohipDataType(cuDT, dataType);
}

hipdnnStatus_t
hipdnnCreateDropoutDescriptor(hipdnnDropoutDescriptor_t *dropoutDesc) {
    return cudnnTohipdnnStatus(
        cudnnCreateDropoutDescriptor((cudnnDropoutDescriptor_t *)dropoutDesc));
}

hipdnnStatus_t hipdnnDropoutGetStatesSize(hipdnnHandle_t handle,
                                          size_t *sizeInBytes) {
    return cudnnTohipdnnStatus(
        cudnnDropoutGetStatesSize((cudnnHandle_t)handle, sizeInBytes));
}

hipdnnStatus_t hipdnnSetDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc,
                                          hipdnnHandle_t handle, float dropout,
                                          void *states, size_t stateSizeInBytes,
                                          unsigned long long seed) {
    return cudnnTohipdnnStatus(cudnnSetDropoutDescriptor(
        (cudnnDropoutDescriptor_t)dropoutDesc, (cudnnHandle_t)handle, dropout,
        states, stateSizeInBytes, seed));
}

hipdnnStatus_t
hipdnnDestroyDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc) {
    return cudnnTohipdnnStatus(
        cudnnDestroyDropoutDescriptor((cudnnDropoutDescriptor_t)dropoutDesc));
}

hipdnnStatus_t
hipdnnSetFilterNdDescriptor(hipdnnFilterDescriptor_t filterDesc,
                            hipdnnDataType_t dataType, // image data type
                            hipdnnTensorFormat_t format, int nbDims,
                            const int filterDimA[]) {
    cudnnDataType_t cuDT;
    cudnnTensorFormat_t cuTF;
    hipdnnStatus_t retval;
    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnTensorFormat(format, &cuTF);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnSetFilterNdDescriptor(
        (cudnnFilterDescriptor_t)filterDesc, cuDT, cuTF, nbDims, filterDimA));
}

hipdnnStatus_t hipdnnGetFilterNdDescriptor(
    const hipdnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    hipdnnDataType_t *dataType, // image data type
    hipdnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
    cudnnDataType_t cuDT;
    cudnnTensorFormat_t cuTF;
    hipdnnStatus_t retval;
    retval = cudnnTohipdnnStatus(cudnnGetFilterNdDescriptor(
        (cudnnFilterDescriptor_t)filterDesc, nbDimsRequested, &cuDT, &cuTF,
        nbDims, filterDimA));
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = cudnnTohipTensorFormat(cuTF, format);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipDataType(cuDT, dataType);
}

hipdnnStatus_t
hipdnnDestroyFilterDescriptor(hipdnnFilterDescriptor_t filterDesc) {
    return cudnnTohipdnnStatus(
        cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)filterDesc));
}

hipdnnStatus_t hipdnnSetConvolutionNdDescriptor(
    hipdnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
    const int padA[], const int filterStrideA[], const int dilationA[],
    hipdnnConvolutionMode_t mode,
    hipdnnDataType_t computeType) // convolution data type
{
    cudnnDataType_t cuDT;
    cudnnConvolutionMode_t cuCM;
    hipdnnStatus_t retval;

    cuCM = hipTocudnnConvolutionMode(mode);

    retval = hipTocudnnDataType(computeType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnSetConvolutionNdDescriptor(
        (cudnnConvolutionDescriptor_t)convDesc, arrayLength, padA,
        filterStrideA, dilationA, cuCM, cuDT));
}

hipdnnStatus_t hipdnnSetPoolingNdDescriptor(
    hipdnnPoolingDescriptor_t poolingDesc, const hipdnnPoolingMode_t mode,
    const hipdnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[], const int strideA[]) {
    cudnnPoolingMode_t cuPM;
    cudnnNanPropagation_t cuNP;
    hipdnnStatus_t retval;

    retval = hipTocudnnPoolingMode(mode, &cuPM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnNanPropagation(maxpoolingNanOpt, &cuNP);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnSetPoolingNdDescriptor(
        (cudnnPoolingDescriptor_t)poolingDesc, cuPM, cuNP, nbDims, windowDimA,
        paddingA, strideA));
}

const char *hipdnnGetErrorString(hipdnnStatus_t status) {
    cudnnStatus_t cstatus;
    cstatus = hipdnnTocudnnStatus(status);
    return cudnnGetErrorString(cstatus);
}

// RNN APIs

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t *rnnDesc) {
    return cudnnTohipdnnStatus(
        cudnnCreateRNNDescriptor((cudnnRNNDescriptor_t *)rnnDesc));
}

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc) {
    return cudnnTohipdnnStatus(
        cudnnDestroyRNNDescriptor((cudnnRNNDescriptor_t)rnnDesc));
}

hipdnnStatus_t hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                             const int minibatch,
                                             const hipdnnDataType_t dataType,
                                             hipdnnPersistentRNNPlan_t *plan) {
    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;

    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(
        cudnnCreatePersistentRNNPlan((cudnnRNNDescriptor_t)rnnDesc, minibatch,
                                     cuDT, (cudnnPersistentRNNPlan_t *)plan));
}

hipdnnStatus_t hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                          hipdnnPersistentRNNPlan_t plan) {
    return cudnnTohipdnnStatus(cudnnSetPersistentRNNPlan(
        (cudnnRNNDescriptor_t)rnnDesc, (cudnnPersistentRNNPlan_t)plan));
}

hipdnnStatus_t hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan) {
    return cudnnTohipdnnStatus(
        cudnnDestroyPersistentRNNPlan((cudnnPersistentRNNPlan_t)plan));
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v6(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers,
    hipdnnDropoutDescriptor_t
        dropoutDesc, // Between layers, not between recurrent steps.
    hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
    hipdnnRNNMode_t mode, hipdnnRNNAlgo_t algo, hipdnnDataType_t dataType) {
    cudnnRNNInputMode_t cuRIM;
    cudnnDirectionMode_t cuDM;
    cudnnRNNMode_t cuRM;
    cudnnRNNAlgo_t cuRA;
    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;

    retval = hipTocudnnRNNInputMode(inputMode, &cuRIM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnDirectionMode(direction, &cuDM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnRNNMode(mode, &cuRM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnRNNAlgo(algo, &cuRA);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnSetRNNDescriptor_v6(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, hiddenSize,
        numLayers, (cudnnDropoutDescriptor_t)dropoutDesc, cuRIM, cuDM, cuRM,
        cuRA, cuDT));
}

hipdnnStatus_t hipdnnSetRNNDescriptor(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, int hiddenSize,
    int numLayers,
    hipdnnDropoutDescriptor_t
        dropoutDesc, // Between layers, not between recurrent steps.
    hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
    hipdnnRNNMode_t mode, hipdnnRNNAlgo_t algo, hipdnnDataType_t dataType) {
    cudnnRNNInputMode_t cuRIM;
    cudnnDirectionMode_t cuDM;
    cudnnRNNMode_t cuRM;
    cudnnRNNAlgo_t cuRA;
    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;

    retval = hipTocudnnRNNInputMode(inputMode, &cuRIM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnDirectionMode(direction, &cuDM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnRNNMode(mode, &cuRM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnRNNAlgo(algo, &cuRA);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnSetRNNDescriptor(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, hiddenSize,
        numLayers, (cudnnDropoutDescriptor_t)dropoutDesc, cuRIM, cuDM, cuRM,
        cuRA, cuDT));
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v5(
    hipdnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
    hipdnnDropoutDescriptor_t
        dropoutDesc, /* Between layers, not between recurrent steps. */
    hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
    hipdnnRNNMode_t mode, hipdnnDataType_t dataType) {
    cudnnRNNInputMode_t cuRIM;
    cudnnDirectionMode_t cuDM;
    cudnnRNNMode_t cuRM;
    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;

    retval = hipTocudnnRNNInputMode(inputMode, &cuRIM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnDirectionMode(direction, &cuDM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnRNNMode(mode, &cuRM);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;
    return cudnnTohipdnnStatus(cudnnSetRNNDescriptor_v5(
        (cudnnRNNDescriptor_t)rnnDesc, hiddenSize, numLayers,
        (cudnnDropoutDescriptor_t)dropoutDesc, cuRIM, cuDM, cuRM, cuDT));
}

hipdnnStatus_t hipdnnGetRNNWorkspaceSize(hipdnnHandle_t handle,
                                         const hipdnnRNNDescriptor_t rnnDesc,
                                         const int seqLength,
                                         const hipdnnTensorDescriptor_t *xDesc,
                                         size_t *sizeInBytes) {
    return cudnnTohipdnnStatus(cudnnGetRNNWorkspaceSize(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, seqLength,
        (cudnnTensorDescriptor_t *)xDesc, sizeInBytes));
}

hipdnnStatus_t hipdnnGetRNNTrainingReserveSize(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
    return cudnnTohipdnnStatus(cudnnGetRNNTrainingReserveSize(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, seqLength,
        (cudnnTensorDescriptor_t *)xDesc, sizeInBytes));
}

hipdnnStatus_t hipdnnGetRNNParamsSize(hipdnnHandle_t handle,
                                      const hipdnnRNNDescriptor_t rnnDesc,
                                      const hipdnnTensorDescriptor_t xDesc,
                                      size_t *sizeInBytes,
                                      hipdnnDataType_t dataType) {
    cudnnDataType_t cuDT;
    hipdnnStatus_t retval;

    retval = hipTocudnnDataType(dataType, &cuDT);
    if (retval != HIPDNN_STATUS_SUCCESS)
        return retval;

    return cudnnTohipdnnStatus(cudnnGetRNNParamsSize(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc,
        (cudnnTensorDescriptor_t)xDesc, sizeInBytes, cuDT));
}

hipdnnStatus_t hipdnnGetRNNLinLayerMatrixParams(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc, const int layer,
    const hipdnnTensorDescriptor_t xDesc, const hipdnnFilterDescriptor_t wDesc,
    const void *w, const int linLayerID,
    hipdnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {
    return cudnnTohipdnnStatus(cudnnGetRNNLinLayerMatrixParams(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, layer,
        (cudnnTensorDescriptor_t)xDesc, (cudnnFilterDescriptor_t)wDesc, w,
        linLayerID, (cudnnFilterDescriptor_t)linLayerMatDesc, linLayerMat));
}

hipdnnStatus_t hipdnnGetRNNLinLayerBiasParams(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc, const int layer,
    const hipdnnTensorDescriptor_t xDesc, const hipdnnFilterDescriptor_t wDesc,
    const void *w, const int linLayerID,
    hipdnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {
    return cudnnTohipdnnStatus(cudnnGetRNNLinLayerBiasParams(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, layer,
        (cudnnTensorDescriptor_t)xDesc, (cudnnFilterDescriptor_t)wDesc, w,
        linLayerID, (cudnnFilterDescriptor_t)linLayerBiasDesc, linLayerBias));
}

hipdnnStatus_t hipdnnRNNForwardInference(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc, const void *x,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t cxDesc, const void *cx,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t *yDesc, void *y,
    const hipdnnTensorDescriptor_t hyDesc, void *hy,
    const hipdnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
    return cudnnTohipdnnStatus(cudnnRNNForwardInference(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, seqLength,
        (cudnnTensorDescriptor_t *)xDesc, x, (cudnnTensorDescriptor_t)hxDesc,
        hx, (cudnnTensorDescriptor_t)cxDesc, cx, (cudnnFilterDescriptor_t)wDesc,
        w, (cudnnTensorDescriptor_t *)yDesc, y, (cudnnTensorDescriptor_t)hyDesc,
        hy, (cudnnTensorDescriptor_t)cyDesc, cy, workspace,
        workSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnRNNForwardTraining(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc, const void *x,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t cxDesc, const void *cx,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t *yDesc, void *y,
    const hipdnnTensorDescriptor_t hyDesc, void *hy,
    const hipdnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    return cudnnTohipdnnStatus(cudnnRNNForwardTraining(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, seqLength,
        (cudnnTensorDescriptor_t *)xDesc, x, (cudnnTensorDescriptor_t)hxDesc,
        hx, (cudnnTensorDescriptor_t)cxDesc, cx, (cudnnFilterDescriptor_t)wDesc,
        w, (cudnnTensorDescriptor_t *)yDesc, y, (cudnnTensorDescriptor_t)hyDesc,
        hy, (cudnnTensorDescriptor_t)cyDesc, cy, workspace,
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
}

hipdnnStatus_t
hipdnnRNNBackwardData(hipdnnHandle_t handle,
                      const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
                      const hipdnnTensorDescriptor_t *yDesc, const void *y,
                      const hipdnnTensorDescriptor_t *dyDesc, const void *dy,
                      const hipdnnTensorDescriptor_t dhyDesc, const void *dhy,
                      const hipdnnTensorDescriptor_t dcyDesc, const void *dcy,
                      const hipdnnFilterDescriptor_t wDesc, const void *w,
                      const hipdnnTensorDescriptor_t hxDesc, const void *hx,
                      const hipdnnTensorDescriptor_t cxDesc, const void *cx,
                      const hipdnnTensorDescriptor_t *dxDesc, void *dx,
                      const hipdnnTensorDescriptor_t dhxDesc, void *dhx,
                      const hipdnnTensorDescriptor_t dcxDesc, void *dcx,
                      void *workspace, size_t workSpaceSizeInBytes,
                      void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    return cudnnTohipdnnStatus(cudnnRNNBackwardData(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, seqLength,
        (cudnnTensorDescriptor_t *)yDesc, y, (cudnnTensorDescriptor_t *)dyDesc,
        dy, (cudnnTensorDescriptor_t)dhyDesc, dhy,
        (cudnnTensorDescriptor_t)dcyDesc, dcy, (cudnnFilterDescriptor_t)wDesc,
        w, (cudnnTensorDescriptor_t)hxDesc, hx, (cudnnTensorDescriptor_t)cxDesc,
        cx, (cudnnTensorDescriptor_t *)dxDesc, dx,
        (cudnnTensorDescriptor_t)dhxDesc, dhx, (cudnnTensorDescriptor_t)dcxDesc,
        dcx, workspace, workSpaceSizeInBytes, reserveSpace,
        reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnRNNBackwardWeights(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc, const void *x,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t *yDesc, const void *y, const void *workspace,
    size_t workSpaceSizeInBytes, const hipdnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    return cudnnTohipdnnStatus(cudnnRNNBackwardWeights(
        (cudnnHandle_t)handle, (cudnnRNNDescriptor_t)rnnDesc, seqLength,
        (cudnnTensorDescriptor_t *)xDesc, x, (cudnnTensorDescriptor_t)hxDesc,
        hx, (cudnnTensorDescriptor_t *)yDesc, y, workspace,
        workSpaceSizeInBytes, (cudnnFilterDescriptor_t)dwDesc, dw, reserveSpace,
        reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnBatchNormalizationForwardInference(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode,
    const void *alpha, // alpha[0] = result blend factor
    const void *beta,  // beta[0] = dest layer blend factor
    const hipdnnTensorDescriptor_t xDesc,
    const void *x, // NxCxHxW
    const hipdnnTensorDescriptor_t yDesc,
    void *y, // NxCxHxW
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {

    return cudnnTohipdnnStatus(cudnnBatchNormalizationForwardInference(
        (cudnnHandle_t)handle, hipTocudnnBatchNormMode(mode), alpha, beta,
        (cudnnTensorDescriptor_t)xDesc, x, (cudnnTensorDescriptor_t)yDesc, y,
        (cudnnTensorDescriptor_t)bnScaleBiasMeanVarDesc, bnScale, bnBias,
        estimatedMean, estimatedVariance, epsilon));
}

// CNTK 2.4 SUPPORT

hipdnnStatus_t hipdnnCreateReduceTensorDescriptor(
    hipdnnReduceTensorDescriptor_t *reduceTensorDesc) {
    return cudnnTohipdnnStatus(cudnnCreateReduceTensorDescriptor(
        (cudnnReduceTensorDescriptor_t *)reduceTensorDesc));
}

hipdnnStatus_t
hipdnnSetTensor4dDescriptorEx(hipdnnTensorDescriptor_t tensorDesc,
                              hipdnnDataType_t dataType, /* image data type */
                              int n, /* number of inputs (batch size) */
                              int c, /* number of input feature maps */
                              int h, /* height of input section */
                              int w, /* width of input section */
                              int nStride, int cStride, int hStride,
                              int wStride) {
    cudnnDataType_t cuDT;
    hipdnnStatus_t retVal;
    retVal = hipTocudnnDataType(dataType, &cuDT);
    if (retVal == HIPDNN_STATUS_SUCCESS)
        return cudnnTohipdnnStatus(cudnnSetTensor4dDescriptorEx(
            (cudnnTensorDescriptor_t)tensorDesc, cuDT, n, c, h, w, nStride,
            cStride, hStride, wStride));
    return retVal;
}

hipdnnStatus_t
hipdnnSetReduceTensorDescriptor(hipdnnReduceTensorDescriptor_t reduceTensorDesc,
                                hipdnnReduceTensorOp_t reduceTensorOp,
                                hipdnnDataType_t reduceTensorCompType,
                                hipdnnNanPropagation_t reduceTensorNanOpt,
                                hipdnnReduceTensorIndices_t reduceTensorIndices,
                                hipdnnIndicesType_t reduceTensorIndicesType) {
    cudnnReduceTensorOp_t cuRTO;
    cudnnDataType_t cuDT;
    cudnnNanPropagation_t cuNP;
    cudnnReduceTensorIndices_t cuRTI;
    cudnnIndicesType_t cuIT;
    hipdnnStatus_t retVal;

    retVal = hipTocudnnReduceTensorOp(reduceTensorOp, &cuRTO);
    if (retVal == HIPDNN_STATUS_SUCCESS) {
        retVal = hipTocudnnDataType(reduceTensorCompType, &cuDT);
        if (retVal == HIPDNN_STATUS_SUCCESS) {
            retVal = hipTocudnnNanPropagation(reduceTensorNanOpt, &cuNP);
            if (retVal == HIPDNN_STATUS_SUCCESS) {
                retVal =
                    hipTocudnnReduceTensorIndices(reduceTensorIndices, &cuRTI);
                if (retVal == HIPDNN_STATUS_SUCCESS) {
                    retVal =
                        hipTocudnnIndicesType(reduceTensorIndicesType, &cuIT);
                    if (retVal == HIPDNN_STATUS_SUCCESS) {
                        return cudnnTohipdnnStatus(
                            cudnnSetReduceTensorDescriptor(
                                (cudnnReduceTensorDescriptor_t)reduceTensorDesc,
                                cuRTO, cuDT, cuNP, cuRTI, cuIT));
                    }
                }
            }
        }
    }
    return retVal;
}

hipdnnStatus_t hipdnnGetReductionWorkspaceSize(
    hipdnnHandle_t handle,
    const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
    const hipdnnTensorDescriptor_t aDesc, const hipdnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
    return cudnnTohipdnnStatus(cudnnGetReductionWorkspaceSize(
        (cudnnHandle_t)handle, (cudnnReduceTensorDescriptor_t)reduceTensorDesc,
        (cudnnTensorDescriptor_t)aDesc, (cudnnTensorDescriptor_t)cDesc,
        sizeInBytes));
}

hipdnnStatus_t hipdnnReduceTensor(
    hipdnnHandle_t handle,
    const hipdnnReduceTensorDescriptor_t reduceTensorDesc, void *indices,
    size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const hipdnnTensorDescriptor_t aDesc, const void *A,
    const void *beta, const hipdnnTensorDescriptor_t cDesc, void *C) {
    return cudnnTohipdnnStatus(cudnnReduceTensor(
        (cudnnHandle_t)handle, (cudnnReduceTensorDescriptor_t)reduceTensorDesc,
        indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha,
        (cudnnTensorDescriptor_t)aDesc, A, beta, (cudnnTensorDescriptor_t)cDesc,
        C));
}

hipdnnStatus_t hipdnnDestroyReduceTensorDescriptor(
    hipdnnReduceTensorDescriptor_t reduceTensorDesc) {
    return cudnnTohipdnnStatus(cudnnDestroyReduceTensorDescriptor(
        (cudnnReduceTensorDescriptor_t)reduceTensorDesc));
}

 hipdnnStatus_t hipdnnSetConvolutionGroupCount(
    hipdnnConvolutionDescriptor_t convDesc, int groupCount ) {
    return cudnnTohipdnnStatus(cudnnSetConvolutionGroupCount(
        (cudnnConvolutionDescriptor_t)convDesc, groupCount) );
}

// --- Fusion API ---
// typedef struct fusionConvolutionForwardCreate fusionConvolutionForwardCreate_t;
  typedef struct{
        hipdnnConvolutionDescriptor_t  convDesc;
        hipdnnFilterDescriptor_t       wDesc;
    }fusionConvolutionForwardCreate_t;
typedef struct {
    fusionConvolutionForwardCreate_t     creationParam;
    void                           *alpha;
    /*hipdnnTensorDescriptor_t       xDesc; */ // set at execution call
    /* void                           *x; */ // set at execution call
    /* hipdnnFilterDescriptor_t       wDesc;*/ //set at operator creation
    void                           *w;
    /* hipdnnConvolutionDescriptor_t  convDesc;*/ //set at operator creation
    /*hipdnnConvolutionFwdAlgo_t           algo; */ //calculate at execution call
    /*void                                 *workSpace; */ //calculate at execution call
    /*size_t                               workSpaceSizeInBytes; */
     void                           *beta;
    /* hipdnnTensorDescriptor_t       yDesc; */ // set at execution call
    /*void                                 *y; */ // set at execution call
} fusionConvolutionForwardArgs_t;

// typedef struct fusionActivationForwardCreate fusionActivationForwardCreate_t;
typedef struct {
        hipdnnActivationMode_t activationMode;
} fusionActivationForwardCreate_t;
typedef struct {
 fusionActivationForwardCreate_t       creationParam;
    /*hipdnnActivationDescriptor_t     activationDesc;*/ // set at exection time
     void                         *alpha;
    /* hipdnnTensorDescriptor_t   xDesc; */ // set at execution time
    /* void                       *x; */ // set at execution time
     void                         *beta;
    /* hipdnnTensorDescriptor_t   yDesc;*/ // set at execution time
    /*void                             *y;*/ // set at execution time
    double                             activAlpha;
    double                             activBeta;
    double                             activGamma;
} fusionActivationForwardArgs_t;

// typedef struct fusionBatchNormInferenceCreate fusionBatchNormInferenceCreate_t;
typedef struct {
        hipdnnBatchNormMode_t             bnMode;
         hipdnnTensorDescriptor_t    bnScaleBiasMeanVarDesc;
}fusionBatchNormInferenceCreate_t;
typedef struct {
    fusionBatchNormInferenceCreate_t  creationParam;
    /*hipdnnBatchNormMode_t             mode;*/
     void                        *alpha;
     void                        *beta;
    /* hipdnnTensorDescriptor_t    xDesc;*/ //set at execution
    /* void                        *x;*/ //set at execution
    /* hipdnnTensorDescriptor_t    yDesc;*/ //set at execution
    /*void                              *y;*/ // set at execution
    /* hipdnnTensorDescriptor_t    bnScaleBiasMeanVarDesc;*/
     void                        *bnScale;
     void                        *bnBias;
     void                        *estimatedMean;
     void                        *estimatedVariance;
    double                            epsilon;
} fusionBatchNormInferenceArgs_t;

// typedef struct fusionBiasForwardCreate fusionBiasForwardCreate_t;
typedef struct {
        hipdnnTensorDescriptor_t       biasDesc;
} fusionBiasForwardCreate_t;
typedef struct {
    // No separate bias Forward function in cudnn
    fusionBiasForwardCreate_t            creationParam;
    void                           *alpha; //alpha1
    /* hipdnnTensorDescriptor_t       xDesc;*/ // unused
    /* void                           *x;*/ // unused
    /* hipdnnFilterDescriptor_t       wDesc;*/ //used
    /* void                           *w; */ //unused
    /* hipdnnConvolutionDescriptor_t  convDesc;*/ //unused
    /*hipdnnConvolutionFwdAlgo_t           algo; */ //unused
    /*void                                 *workSpace; */ //unused
    /*size_t                               workSpaceSizeInBytes; */ //unused
    void                           *beta; //alpha2
    /* hipdnnTensorDescriptor_t       zDesc; */ //unused
    /* void                           *z;  */ //unused
    /* hipdnnTensorDescriptor_t       biasDesc;*/ //set at creation time
    void                           *bias;
    /* hipdnnActivationDescriptor_t   activationDesc;*/ //execution time
    /* hipdnnTensorDescriptor_t       yDesc; */ // unused
    /*void                                 *y; */ //unused
} fusionBiasForwardArgs_t;

#define FUSION_MAX 7 // Max Number of layers to be fused in single fusion plan
typedef struct { /* Fusion plan */
    hipdnnHandle_t           handle;
    hipdnnFusionDirection_t  fuseDirection;
    hipdnnTensorDescriptor_t inputDesc;
    int                      fusePlanTime;
    int                      fusePlanId;
    char                     fuseOpSeq[FUSION_MAX];
    int                      fuseOpCount;
    void*                    fuseOpPtrs[FUSION_MAX];
} fusionPlan_t;

typedef struct {
    char                    fuseOpArgsSeq[FUSION_MAX];
    int                      fuseOpArgsCount;
    void*                   fuseOpArgsPtrs[FUSION_MAX];
}fusionOpArgs_t;

int fusionValidate (fusionPlan_t* basePlan, fusionPlan_t* checkPlan) {
    if( basePlan->fusePlanId == checkPlan->fusePlanId) {
        if ( basePlan->fusePlanTime == checkPlan->fusePlanTime){
            return 1;
        }
    }
    return 0;
}

// ============================================================================

hipdnnStatus_t
hipdnnCreateFusionPlan(hipdnnFusionPlanDescriptor_t*  fusePlanDesc,
                       const hipdnnFusionDirection_t  fuseDirection,
                       const hipdnnTensorDescriptor_t inputDesc) {

    *fusePlanDesc = (void*)malloc(sizeof(fusionPlan_t));
    CHECK_MALLOC(*fusePlanDesc);
    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)(*fusePlanDesc);
    fusePlanDesc_cast->fusePlanTime = time(0);
    fusePlanDesc_cast->fusePlanId = rand();
    fusePlanDesc_cast->fuseDirection = fuseDirection;
    fusePlanDesc_cast->inputDesc = inputDesc;
    for (int i=0; i<FUSION_MAX; i++) {
        fusePlanDesc_cast->fuseOpSeq[i] = '\0';
        fusePlanDesc_cast->fuseOpPtrs[i] = '\0';

    }
    fusePlanDesc_cast->fuseOpCount= 0;
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
hipdnnCreateOpConvForward(hipdnnFusionPlanDescriptor_t    fusePlanDesc,
                          hipdnnFusionOpDescriptor_t*     convOp,
                          hipdnnConvolutionDescriptor_t   convDesc,
                          const hipdnnTensorDescriptor_t  wDesc ) {

    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    fusePlanDesc_cast->fuseOpCount += 1;
    int newCount = fusePlanDesc_cast->fuseOpCount;
    fusePlanDesc_cast->fuseOpSeq[newCount-1] = 'C';

    *convOp = (void*)malloc(sizeof(fusionConvolutionForwardCreate_t));
    CHECK_MALLOC(*convOp);
    fusionConvolutionForwardCreate_t* convOp_cast = (fusionConvolutionForwardCreate_t*)(*convOp);
    convOp_cast->convDesc=(hipdnnConvolutionDescriptor_t)convDesc;
    convOp_cast->wDesc=(hipdnnTensorDescriptor_t)wDesc;
    fusePlanDesc_cast->fuseOpPtrs[newCount-1] = *convOp;

    return HIPDNN_STATUS_SUCCESS;

}

hipdnnStatus_t
hipdnnCreateOpBiasForward(hipdnnFusionPlanDescriptor_t fusePlanDesc,
                          hipdnnFusionOpDescriptor_t *biasOp,
                          const hipdnnTensorDescriptor_t bDesc) {

    // No separate bias Forward function in cudnn
    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    fusePlanDesc_cast->fuseOpCount += 1;
    int newCount = fusePlanDesc_cast->fuseOpCount;
    fusePlanDesc_cast->fuseOpSeq[newCount-1] = 'B';

    *biasOp = (void*)malloc(sizeof(fusionBiasForwardCreate_t));
    CHECK_MALLOC(*biasOp);
    fusionBiasForwardCreate_t* biasOp_cast = (fusionBiasForwardCreate_t*)(*biasOp);
    biasOp_cast->biasDesc=(hipdnnTensorDescriptor_t)bDesc;
    fusePlanDesc_cast->fuseOpPtrs[newCount-1] = *biasOp;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
hipdnnCreateOpActivationForward(hipdnnFusionPlanDescriptor_t fusePlanDesc,
                                hipdnnFusionOpDescriptor_t *activOp,
                                hipdnnActivationMode_t mode) {


    // No separate bias Forward function in cudnn
    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    fusePlanDesc_cast->fuseOpCount += 1;
    int newCount = fusePlanDesc_cast->fuseOpCount;
    fusePlanDesc_cast->fuseOpSeq[newCount-1] = 'A';

    *activOp = (void*)malloc(sizeof(fusionActivationForwardCreate_t));
    CHECK_MALLOC(*activOp);
    fusionActivationForwardCreate_t* activOp_cast = (fusionActivationForwardCreate_t*)(*activOp);
    activOp_cast->activationMode = (hipdnnActivationMode_t)mode; // decriptor will be created on execution based on this
    fusePlanDesc_cast->fuseOpPtrs[newCount-1] = *activOp;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnCreateOpBatchNormInference(
    hipdnnFusionPlanDescriptor_t fusePlanDesc, hipdnnFusionOpDescriptor_t *bnOp,
    const hipdnnBatchNormMode_t bn_mode,
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc) {

    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    fusePlanDesc_cast->fuseOpCount += 1;
    int newCount = fusePlanDesc_cast->fuseOpCount;
    fusePlanDesc_cast->fuseOpSeq[newCount-1] = 'N';

    *bnOp = (void*)malloc(sizeof(fusionBatchNormInferenceCreate_t));
    CHECK_MALLOC(*bnOp);
    fusionBatchNormInferenceCreate_t* bnOp_cast = (fusionBatchNormInferenceCreate_t*)(*bnOp);
    bnOp_cast->bnMode=(hipdnnBatchNormMode_t)bn_mode;
    bnOp_cast->bnScaleBiasMeanVarDesc=(hipdnnTensorDescriptor_t)bnScaleBiasMeanVarDesc;
    fusePlanDesc_cast->fuseOpPtrs[newCount-1] = *bnOp;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
hipdnnCompileFusionPlan(hipdnnHandle_t handle,
                        hipdnnFusionPlanDescriptor_t fusePlanDesc) {

    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    fusePlanDesc_cast->handle = handle;
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnFusionPlanGetOp(hipdnnFusionPlanDescriptor_t fusePlanDesc,
                                     const int op_idx,
                                     hipdnnFusionOpDescriptor_t *op) {
    hipdnnStatus_t retVal;
    if( ((fusionPlan_t*)fusePlanDesc)->fuseOpCount > op_idx ) {
        *op = ((fusionPlan_t*)fusePlanDesc)->fuseOpPtrs[op_idx];
        retVal = HIPDNN_STATUS_SUCCESS;
    }
    else {
        retVal = HIPDNN_STATUS_INVALID_VALUE;
    }
    return retVal;
}

hipdnnStatus_t hipdnnFusionPlanGetWorkSpaceSize(
    hipdnnHandle_t handle, hipdnnFusionPlanDescriptor_t fusePlanDesc,
    size_t *workSpaceSize, hipdnnConvolutionFwdAlgo_t algo) {

    hipdnnStatus_t retVal;
    int convFlag = 0 ;
    int convId = 0;
    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    for( int convId=0; convId < fusePlanDesc_cast->fuseOpCount; convId++ ) {
        if (fusePlanDesc_cast->fuseOpSeq[convId] == 'C') {
            convFlag = 1;
            break;
        }
    }

    if (convFlag==1){
        hipdnnHandle_t handle = fusePlanDesc_cast->handle;
        hipdnnTensorDescriptor_t xDesc = fusePlanDesc_cast->inputDesc;
        fusionConvolutionForwardCreate_t* fuseOpPtrsDesc_cast =
            (fusionConvolutionForwardCreate_t*)(fusePlanDesc_cast->fuseOpPtrs[convId]);
        hipdnnFilterDescriptor_t wDesc = fuseOpPtrsDesc_cast->wDesc;
        hipdnnConvolutionDescriptor_t convDesc = fuseOpPtrsDesc_cast->convDesc;
        hipdnnTensorDescriptor_t yDesc;

        CHECK_HIPDNN(hipdnnCreateTensorDescriptor(&yDesc));
        int n, c, h, w ;
        CHECK_HIPDNN(hipdnnGetConvolution2dForwardOutputDim(convDesc,
            xDesc, wDesc, &n, &c, &h, &w));
        hipdnnDataType_t dataType;
        int temp; // temp is passed for unncessary information
        CHECK_HIPDNN(hipdnnGetTensor4dDescriptor(xDesc, &dataType, &temp,
            &temp, &temp, &temp, &temp, &temp, &temp, &temp));
        hipdnnTensorFormat_t format = HIPDNN_TENSOR_NCHW;
        CHECK_HIPDNN(hipdnnSetTensor4dDescriptor(yDesc, format, dataType,
            n, c, h, w));
        size_t workSpaceSizeInBytes;
        CHECK_HIPDNN(hipdnnGetConvolutionForwardWorkspaceSize( handle, xDesc,
            wDesc, convDesc, yDesc, algo, &workSpaceSizeInBytes));
        retVal = HIPDNN_STATUS_SUCCESS;
    }
    else {
        retVal = HIPDNN_STATUS_NOT_INITIALIZED;
    }
    return retVal;
}

hipdnnStatus_t hipdnnFusionPlanConvolutionGetAlgo(
    hipdnnFusionPlanDescriptor_t fusePlanDesc, const int requestAlgoCount,
    int* returnedAlgoCount, hipdnnConvolutionFwdAlgo_t* returnedAlgos) {

    int convFlag = 0 ;
    int convId =0;
    hipdnnStatus_t retVal;
    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    for( int convId=0; convId < fusePlanDesc_cast->fuseOpCount; convId++ ) {
        if (fusePlanDesc_cast->fuseOpSeq[convId] == 'C') {
            convFlag = 1;
            break;
        }
    }

    if (convFlag==1){
        hipdnnHandle_t handle = fusePlanDesc_cast->handle;
        hipdnnTensorDescriptor_t xDesc = fusePlanDesc_cast->inputDesc;
        fusionConvolutionForwardCreate_t* fuseOpPtrsDesc_cast =
            (fusionConvolutionForwardCreate_t*)(fusePlanDesc_cast->fuseOpPtrs[convId]);
        hipdnnFilterDescriptor_t wDesc = fuseOpPtrsDesc_cast->wDesc;
        hipdnnConvolutionDescriptor_t convDesc = fuseOpPtrsDesc_cast->convDesc;
        hipdnnTensorDescriptor_t yDesc;

        CHECK_HIPDNN(hipdnnCreateTensorDescriptor(&yDesc));
        int n, c, h, w ;
        CHECK_HIPDNN(hipdnnGetConvolution2dForwardOutputDim(convDesc, xDesc,
            wDesc, &n, &c, &h, &w));
        hipdnnDataType_t dataType;
        int temp; // temp is passed for unncessary information
        CHECK_HIPDNN(hipdnnGetTensor4dDescriptor(xDesc, &dataType, &temp, &temp,
            &temp, &temp, &temp, &temp, &temp, &temp));
        hipdnnTensorFormat_t format = HIPDNN_TENSOR_NCHW;
        CHECK_HIPDNN(hipdnnSetTensor4dDescriptor(yDesc, format, dataType,
            n, c, h, w));

        hipdnnConvolutionFwdAlgoPerf_t perfResults;
        CHECK_HIPDNN(hipdnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc,
            convDesc, yDesc, requestAlgoCount, returnedAlgoCount, &perfResults));
        retVal = HIPDNN_STATUS_SUCCESS;
    }
    else {
        retVal = HIPDNN_STATUS_NOT_INITIALIZED;
    }

    return retVal;
}

hipdnnStatus_t hipdnnCreateOperatorArgs(hipdnnOperatorArgs_t* args) {

    *args = malloc(sizeof(fusionOpArgs_t));
    CHECK_MALLOC(*args);
    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)(*args);
    args_cast->fuseOpArgsCount =0;
    for(int i=0; i<FUSION_MAX; i++) {
        args_cast->fuseOpArgsSeq[i]='\0';
        args_cast->fuseOpArgsPtrs[i]='\0';
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
hipdnnSetOpArgsConvForward(hipdnnOperatorArgs_t args,
                           const hipdnnFusionOpDescriptor_t convOp,
                           const void *alpha, const void *beta, const void *w) {

    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)args;
    args_cast->fuseOpArgsCount += 1;
    int newCount = args_cast->fuseOpArgsCount;
    args_cast->fuseOpArgsSeq[newCount-1]='C';

    fusionConvolutionForwardArgs_t* convOpArgs_cast =(fusionConvolutionForwardArgs_t*)malloc(sizeof(fusionConvolutionForwardArgs_t));
    fusionConvolutionForwardCreate_t* convOp_cast = (fusionConvolutionForwardCreate_t*)convOp;
    convOpArgs_cast->creationParam.convDesc = (hipdnnConvolutionDescriptor_t)(convOp_cast->convDesc);
    convOpArgs_cast->creationParam.wDesc = (hipdnnFilterDescriptor_t)(convOp_cast->wDesc);
    convOpArgs_cast->alpha = (void*)alpha;
    convOpArgs_cast->beta = (void*)beta;
    convOpArgs_cast->w = (void*)w;
    args_cast->fuseOpArgsPtrs[newCount-1] = (void*)convOpArgs_cast;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetOpArgsBiasForward(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t biasOp,
    const void *alpha, const void *beta, const void *bias) {

    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)args;
    args_cast->fuseOpArgsCount += 1;
    int newCount = args_cast->fuseOpArgsCount;
    args_cast->fuseOpArgsSeq[newCount-1]='B';

    fusionBiasForwardArgs_t* biasOpArgs_cast =(fusionBiasForwardArgs_t*)malloc(sizeof(fusionBiasForwardArgs_t));
    CHECK_MALLOC(biasOpArgs_cast);
    fusionBiasForwardCreate_t* biasOp_cast = (fusionBiasForwardCreate_t*)biasOp;
    biasOpArgs_cast->creationParam.biasDesc = (hipdnnTensorDescriptor_t)(biasOp_cast->biasDesc);
    biasOpArgs_cast->alpha = (void*)alpha;
    biasOpArgs_cast->beta = (void*)beta;
    biasOpArgs_cast->bias = (void*)bias;
    args_cast->fuseOpArgsPtrs[newCount-1] = (void*)biasOpArgs_cast;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetOpArgsActivForward(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t activOp,
    const void *alpha, const void *beta, double activAlpha, double activBeta,
    double activGamma) {

    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)args;
    args_cast->fuseOpArgsCount += 1;
    int newCount = args_cast->fuseOpArgsCount;
    args_cast->fuseOpArgsSeq[newCount-1]='A';

    fusionActivationForwardArgs_t* activOpArgs_cast =(fusionActivationForwardArgs_t*)malloc(sizeof(fusionActivationForwardArgs_t));
    fusionActivationForwardCreate_t* activOp_cast = (fusionActivationForwardCreate_t*)activOp;
    activOpArgs_cast->creationParam.activationMode = (hipdnnActivationMode_t)(activOp_cast->activationMode);
    activOpArgs_cast->alpha = (void*)alpha;
    activOpArgs_cast->beta = (void*)beta;
    activOpArgs_cast->activAlpha = (double)activAlpha;
    activOpArgs_cast->activBeta = (double)activBeta;
    activOpArgs_cast->activGamma = (double)activGamma;
    args_cast->fuseOpArgsPtrs[newCount-1] = (void*)activOpArgs_cast;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetOpArgsBatchNormInference(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t bnOp,
    const void* alpha, const void* beta, const void* bnScale,
    const void* bnBias, const void* estimatedMean,
    const void* estimatedVariance, double epsilon) {

    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)args;
    args_cast->fuseOpArgsCount += 1;
    int newCount = args_cast->fuseOpArgsCount;
    args_cast->fuseOpArgsSeq[newCount-1]='N';

    fusionBatchNormInferenceArgs_t* bnOpArgs_cast =(fusionBatchNormInferenceArgs_t*) malloc(sizeof(fusionBiasForwardArgs_t));
    fusionBatchNormInferenceCreate_t* bnOp_cast = (fusionBatchNormInferenceCreate_t*)bnOp;
    bnOpArgs_cast->creationParam.bnMode = (hipdnnBatchNormMode_t)(bnOp_cast->bnMode);
    bnOpArgs_cast->creationParam.bnScaleBiasMeanVarDesc = (hipdnnTensorDescriptor_t)(bnOp_cast->bnScaleBiasMeanVarDesc);
    bnOpArgs_cast->alpha = (void*)alpha;
    bnOpArgs_cast->beta = (void*)beta;
    bnOpArgs_cast->bnScale = (void*)bnScale;
    bnOpArgs_cast->bnBias = (void*)bnBias;
    bnOpArgs_cast->estimatedMean = (void*)estimatedMean;
    bnOpArgs_cast->estimatedVariance = (void*)estimatedVariance;
    bnOpArgs_cast->epsilon = (double)epsilon;
    args_cast->fuseOpArgsPtrs[newCount-1] = (void*)bnOpArgs_cast;

    return HIPDNN_STATUS_SUCCESS;
}
//++++++++++++++++++++++++++++++++++++++++++++
#include <vector>
#include <iomanip>
void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer ( 1 << 20 );
  hipMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        hipMemcpyDeviceToHost);
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << n << ", c=" << c << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << "\t"<<  std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}
//++++++++++++++++++++++++++++++++++++++++++++
hipdnnStatus_t
hipdnnExecuteFusionPlan(const hipdnnHandle_t handle,
                        const hipdnnFusionPlanDescriptor_t fusePlanDesc,
                        const hipdnnTensorDescriptor_t inputDesc,
                        const void *input,
                        const hipdnnTensorDescriptor_t outputDesc, void *output,
                        hipdnnOperatorArgs_t args) {

    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)args;
    // Sanity Checks
    if (fusePlanDesc_cast->fuseOpCount != args_cast->fuseOpArgsCount) {
        return HIPDNN_STATUS_INVALID_VALUE;
    }

    const void* curInput = input;
    hipdnnTensorDescriptor_t curInputDesc = inputDesc;
    for( int Id=0; Id < fusePlanDesc_cast->fuseOpCount; Id++ ) {
        // Convolution
        if (fusePlanDesc_cast->fuseOpSeq[Id] == 'C') {
            fusionConvolutionForwardArgs_t* convArgs_cast;
            for( int convId=0; convId < args_cast->fuseOpArgsCount; convId++ ) {
                if (args_cast->fuseOpArgsSeq[convId] == 'C') {
                    convArgs_cast = (fusionConvolutionForwardArgs_t*)(args_cast->fuseOpArgsPtrs[convId]);
                    args_cast->fuseOpArgsSeq[convId]='\0'; break;
                }
            }
            hipdnnHandle_t handle = fusePlanDesc_cast->handle;
            hipdnnFilterDescriptor_t filterDesc = (convArgs_cast->creationParam).wDesc;
            void* filter = convArgs_cast->w;
            hipdnnConvolutionDescriptor_t convDesc = (convArgs_cast->creationParam).convDesc;

            hipdnnConvolutionFwdAlgo_t algo;
            int retAlgoCount;
            hipdnnConvolutionFwdAlgoPerf_t algoPerf[5];
            void* workSpace;
            size_t workSpaceSizeInBytes;
            CHECK_HIPDNN(hipdnnFindConvolutionForwardAlgorithmEx(handle,
                curInputDesc, curInput, filterDesc, filter, convDesc, outputDesc,
                output, 5 /*MaxAlgoCount*/, &retAlgoCount, algoPerf, workSpace,
                workSpaceSizeInBytes));
            algo = algoPerf[0].algo;
            // hipdnnConvolutionFwdPreference_t preference = HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST;
            // CHECK_HIPDNN(hipdnnGetConvolutionForwardAlgorithm( handle,
            //     curInputDesc, filterDesc, convDesc, outDesc, preference,
            //     0 /*memoryLimitInBytes*/ ,&algo));
            CHECK_HIPDNN(hipdnnGetConvolutionForwardWorkspaceSize( handle, curInputDesc,
                filterDesc, convDesc, outputDesc, algo, &workSpaceSizeInBytes));
            CHECK_HIP(hipMalloc(&workSpace, workSpaceSizeInBytes));

          //+++++++++++++++ Conv Fwd call +++++++++++++++++++++
              std::cout << "Handle Inside:" <<handle<<std::endl;
              hipdnnDataType_t dataType;
              int ni,ci,hi,wi,sni,sci,shi,swi;
              CHECK_HIPDNN(hipdnnGetTensor4dDescriptor(curInputDesc, &dataType,
                  &ni,&ci,&hi,&wi,&sni,&sci,&shi,&swi));
              std::cout << "in_data:" << std::endl;
              print((float*)curInput,ni,ci,hi,wi);

              hipdnnDataType_t ftype;
              hipdnnTensorFormat_t fformat;
              int filtDim[4]; int nbDim;
              CHECK_HIPDNN(hipdnnGetFilterNdDescriptor(filterDesc, 4, &ftype, &fformat,
                    &nbDim, filtDim));
              std::cout << "filter_data:" << std::endl;
              print((float*)filter, filtDim[0],filtDim[1],filtDim[2],filtDim[3]);

              int phc,pwc,uc,vc,xc,yc;
              hipdnnConvolutionMode_t modec;
              hipdnnDataType_t cctype;
              CHECK_HIPDNN(hipdnnGetConvolution2dDescriptor(convDesc,&phc,&pwc,&uc,&vc,&xc,&yc,&modec,&cctype));
              std::cout<< "Conv-Exec:"<<convDesc<<std::endl;
                std::cout << "pad_h: " << phc << std::endl;
                std::cout << "pad_w: " << pwc << std::endl;
                std::cout << "str_h: " << uc << std::endl;
                std::cout << "str_w: " << vc << std::endl;
                std::cout << "dil_h: " << xc << std::endl;
                std::cout << "dil_w: " << yc << std::endl;
                std::cout << "mode:" << modec << std::endl;
                std::cout << "Type:" << cctype << std::endl;
                std::cout << "alpha:" << *((float*)(convArgs_cast->alpha)) << std::endl;
                std::cout << "beta:" << *((float*)(convArgs_cast->beta)) << std::endl;
                std::cout << std::endl;
                std::cout << "Workspace size: " <<workSpaceSizeInBytes  << std::endl;
                std::cout<<"ConvAlgorithm:"<<algo<<std::endl;
                std::cout << std::endl;
            //++++++++++++++++++++++++++++++++++++++++++++++++++++++

            CHECK_HIPDNN(hipdnnConvolutionForward( handle, convArgs_cast->alpha,
                 curInputDesc, curInput, filterDesc, filter, convDesc, algo,
                 workSpace, workSpaceSizeInBytes, convArgs_cast->beta, outputDesc, output));
            curInput = output;
            curInputDesc = outputDesc;
            CHECK_HIP(hipFree(workSpace));

        }
        // Bias
        else if (fusePlanDesc_cast->fuseOpSeq[Id] == 'B') {
            fusionBiasForwardArgs_t* biasArgs_cast;
            for( int biasId=0; biasId < args_cast->fuseOpArgsCount; biasId++ ) {
                if (args_cast->fuseOpArgsSeq[biasId] == 'B') {
                    biasArgs_cast = (fusionBiasForwardArgs_t*)(args_cast->fuseOpArgsPtrs[biasId]);
                    args_cast->fuseOpArgsSeq[biasId]='\0'; break;
                }
            }

            hipdnnHandle_t handle =  fusePlanDesc_cast->handle;
            hipdnnTensorDescriptor_t biasDesc = (biasArgs_cast->creationParam).biasDesc;
            void* bias = biasArgs_cast->bias;
            CHECK_HIPDNN(hipdnnAddTensor( handle, biasArgs_cast->alpha,
                biasDesc, bias,  biasArgs_cast->beta,
                outputDesc, (void*)curInput /*Inplace add*/));

        }
        // Activation
        else if (fusePlanDesc_cast->fuseOpSeq[Id] == 'A') {
            fusionActivationForwardArgs_t* activArgs_cast;
            for( int activId=0; activId < args_cast->fuseOpArgsCount; activId++ ) {
                if (args_cast->fuseOpArgsSeq[activId] == 'A') {
                    activArgs_cast = (fusionActivationForwardArgs_t*)(args_cast->fuseOpArgsPtrs[activId]);
                    args_cast->fuseOpArgsSeq[activId]='\0'; break;
                }
            }

            hipdnnActivationDescriptor_t activationDesc;
            CHECK_HIPDNN(hipdnnCreateActivationDescriptor(&activationDesc));
            hipdnnActivationMode_t activMode = (activArgs_cast->creationParam).activationMode;
            hipdnnNanPropagation_t reluNanOpt = HIPDNN_PROPAGATE_NAN;
            CHECK_HIPDNN(hipdnnSetActivationDescriptor( activationDesc, activMode,
                reluNanOpt, activArgs_cast->activAlpha, activArgs_cast->activBeta,
                activArgs_cast->activGamma));

            CHECK_HIPDNN(hipdnnActivationForward( fusePlanDesc_cast->handle,
                activationDesc, activArgs_cast->alpha, curInputDesc, curInput,
                activArgs_cast->beta, outputDesc, output));
                curInput = output;
                curInputDesc = outputDesc;

        }
        // Batch Norm
        else if (fusePlanDesc_cast->fuseOpSeq[Id] == 'N') {
            fusionBatchNormInferenceArgs_t* normArgs_cast;
            for( int normId=0; normId < args_cast->fuseOpArgsCount; normId++ ) {
                if (args_cast->fuseOpArgsSeq[normId] == 'C') {
                    normArgs_cast = (fusionBatchNormInferenceArgs_t*)(args_cast->fuseOpArgsPtrs[normId]);
                    args_cast->fuseOpArgsSeq[normId]='\0'; break;
                }
            }
            hipdnnBatchNormMode_t bnMode = (normArgs_cast->creationParam).bnMode;
            hipdnnTensorDescriptor_t bnDesc = normArgs_cast->creationParam.bnScaleBiasMeanVarDesc;
            CHECK_HIPDNN(hipdnnnBatchNormalizationForwardInference( fusePlanDesc_cast->handle,
                bnMode, normArgs_cast->alpha, normArgs_cast->beta,
                inputDesc, curInput, outputDesc, output, bnDesc, normArgs_cast->bnScale,
                normArgs_cast->bnBias, normArgs_cast->estimatedMean,
                normArgs_cast->estimatedVariance, normArgs_cast->epsilon));
            curInput = output;
        }
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
hipdnnDestroyOperatorArgs(hipdnnOperatorArgs_t args) {

    fusionOpArgs_t* args_cast = (fusionOpArgs_t*)(args);
    for (int i=0; i<args_cast->fuseOpArgsCount;i++) {
        free(args_cast->fuseOpArgsPtrs[i]);
    }
    free(args_cast);
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t
hipdnnDestroyFusionPlan(hipdnnFusionPlanDescriptor_t fusePlanDesc) {

    fusionPlan_t* fusePlanDesc_cast = (fusionPlan_t*)fusePlanDesc;
    for (int i=0; i<fusePlanDesc_cast->fuseOpCount; i++) {
        free(fusePlanDesc_cast->fuseOpPtrs[i]);
    }
    free(fusePlanDesc);
    return HIPDNN_STATUS_SUCCESS;
}
