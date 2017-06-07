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

#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>



typedef cudnnTensorDescriptor_t hipdnnTensorDescriptor_t;

typedef cudnnFilterDescriptor_t hipdnnFilterDescriptor_t;

typedef cudnnConvolutionDescriptor_t hipdnnConvolutionDescriptor_t;

typedef cudnnLRNDescriptor_t hipdnnLRNDescriptor_t;

typedef cudnnActivationDescriptor_t hipdnnActivationDescriptor_t;

typedef cudnnPoolingDescriptor_t hipdnnPoolingDescriptor_t;

typedef cudnnConvolutionFwdAlgoPerf_t hipdnnConvolutionFwdAlgoPerf_t;

typedef cudnnConvolutionBwdDataAlgoPerf_t hipdnnConvolutionBwdDataAlgoPerf_t;

typedef cudnnConvolutionBwdFilterAlgoPerf_t hipdnnConvolutionBwdFilterAlgoPerf_t;

typedef cudnnOpTensorDescriptor_t hipdnnOpTensorDescriptor_t;



typedef cudaStream_t hipdnnStream_t;


typedef cudnnHandle_t hipdnnHandle_t;
