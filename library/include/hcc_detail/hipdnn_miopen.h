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
#pragma once

#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>

#define HIPDNN_BN_MIN_EPSILON 1e-05

// Defining structure for containing performance and algorithm results

struct hipdnnConvolutionFwdAlgoPerf_t {
    hipdnnConvolutionFwdAlgo_t algo;
    hipdnnStatus_t status;
    float time;
    size_t memory;
    long reserved[3];
};

struct hipdnnConvolutionBwdDataAlgoPerf_t {
    hipdnnConvolutionBwdDataAlgo_t algo;
    hipdnnStatus_t status;
    float time;
    size_t memory;
    long reserved[3];
};
struct hipdnnConvolutionBwdFilterAlgoPerf_t {
    hipdnnConvolutionBwdFilterAlgo_t algo;
    hipdnnStatus_t status;
    float time;
    size_t memory;
    long reserved[3];
};

typedef miopenTensorDescriptor_t hipdnnTensorDescriptor_t;

typedef miopenTensorDescriptor_t hipdnnReduceTensorDescriptor_t;

typedef miopenTensorDescriptor_t hipdnnFilterDescriptor_t;

typedef miopenConvolutionDescriptor_t hipdnnConvolutionDescriptor_t;

typedef miopenLRNDescriptor_t hipdnnLRNDescriptor_t;

typedef miopenActivationDescriptor_t hipdnnActivationDescriptor_t;

typedef miopenPoolingDescriptor_t hipdnnPoolingDescriptor_t;

// HGSOS cudnn makes a distinction
typedef miopenTensorOp_t hipdnnOpTensorDescriptor_t;

typedef miopenAcceleratorQueue_t hipdnnStream_t;

typedef miopenHandle_t hipdnnHandle_t;

typedef miopenTensorOp_t *hipdnnDropoutDescriptor_t;

typedef miopenRNNDescriptor_t hipdnnRNNDescriptor_t;

typedef miopenTensorOp_t *hipdnnPersistentRNNPlan_t;

hipdnnStatus_t miopenTohipConvolutionFwdAlgo(miopenConvFwdAlgorithm_t in,
                                             hipdnnConvolutionFwdAlgo_t *out);

hipdnnStatus_t
miopenTohipConvolutionBwdFilterAlgo(miopenConvBwdWeightsAlgorithm_t in,
                                    hipdnnConvolutionBwdFilterAlgo_t *out);

hipdnnStatus_t
miopenTohipConvolutionBwdDataAlgo(miopenConvBwdDataAlgorithm_t in,
                                  hipdnnConvolutionBwdDataAlgo_t *out);
