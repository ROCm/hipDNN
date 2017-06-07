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


#include <hipDNN.h>



hipdnnStatus_t cudnnTohipdnnStatus(cudnnStatus_t  cStatus)
{
    hipdnnStatus_t retVal;
    switch(cStatus)
    {
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

hipdnnStatus_t hipTocudnnDataType(hipdnnDataType_t in, cudnnDataType_t* out)
{
    switch(in)
    {
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

hipdnnStatus_t cudnnTohipDataType(cudnnDataType_t in, hipdnnDataType_t* out)
{
    switch(in)
    {
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

hipdnnStatus_t cudnnTohipOpTensorOp(cudnnOpTensorOp_t in, hipdnnOpTensorOp_t* out)
{
    switch(in)
    {
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

hipdnnStatus_t hipTocudnnOpTensorOp(hipdnnOpTensorOp_t in, cudnnOpTensorOp_t* out)
{
    switch(in)
    {
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

hipdnnConvolutionMode_t cudnnTohipConvolutionMode( cudnnConvolutionMode_t in )
{
    if( in == CUDNN_CONVOLUTION )
        return HIPDNN_CONVOLUTION;
    else if( in == CUDNN_CROSS_CORRELATION )
        return HIPDNN_CROSS_CORRELATION;
    
    return HIPDNN_CONVOLUTION;
}

cudnnConvolutionMode_t hipTocudnnConvolutionMode( hipdnnConvolutionMode_t in )
{
    if( in == HIPDNN_CONVOLUTION )
        return CUDNN_CONVOLUTION;
    else if( in == HIPDNN_CROSS_CORRELATION )
        return CUDNN_CROSS_CORRELATION;
    
    return CUDNN_CONVOLUTION;
}


//=============================================================================

hipdnnStatus_t  hipTocudnnPoolingMode(    hipdnnPoolingMode_t in, 
                                        cudnnPoolingMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t  cudnnTohipPoolingMode(    cudnnPoolingMode_t in, 
                                        hipdnnPoolingMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case CUDNN_POOLING_MAX:
        *out = HIPDNN_POOLING_MAX;
        break;
    case CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
        *out = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
    case CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING :
        *out = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        break;
    case CUDNN_POOLING_MAX_DETERMINISTIC :
        *out = HIPDNN_POOLING_MAX_DETERMINISTIC;
        break;
    }
    return retVal;
} 


//===================================

hipdnnStatus_t  hipTocudnnLRNMode(    hipdnnLRNMode_t in, 
                                    cudnnLRNMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_LRN_WITHIN_CHANNEL:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED ;
        break;
    case HIPDNN_LRN_CROSS_CHANNEL:
        *out = CUDNN_LRN_CROSS_CHANNEL_DIM1;
        break;
    }
    return retVal;
} 

//=============================================================================


cudnnBatchNormMode_t hipTocudnnBatchNormMode( hipdnnBatchNormMode_t in)
{
    if( in == HIPDNN_BATCHNORM_PER_ACTIVATION )
        return CUDNN_BATCHNORM_PER_ACTIVATION;
    else if( in == HIPDNN_BATCHNORM_SPATIAL )
        return CUDNN_BATCHNORM_SPATIAL;
    
    return CUDNN_BATCHNORM_PER_ACTIVATION;
}


//================================================

hipdnnStatus_t  hipTocudnnActivationMode(hipdnnActivationMode_t in, 
                                        cudnnActivationMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    
    switch(in)
    {
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

hipdnnStatus_t  cudnnTohipActivationMode(cudnnActivationMode_t  in, 
                                        hipdnnActivationMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    
    switch(in)
    {
    case CUDNN_ACTIVATION_SIGMOID :
        *out = HIPDNN_ACTIVATION_SIGMOID;
        break;
    
    case CUDNN_ACTIVATION_RELU :
        *out = HIPDNN_ACTIVATION_RELU;
        break;
    
    case CUDNN_ACTIVATION_TANH :
        *out = HIPDNN_ACTIVATION_TANH;
        break;
    
    case CUDNN_ACTIVATION_CLIPPED_RELU :
        *out = HIPDNN_ACTIVATION_CLIPPED_RELU;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}


//==========================================================
hipdnnStatus_t  hipTocudnnConvolutionFwdAlgo(hipdnnConvolutionFwdAlgo_t in, 
                                         cudnnConvolutionFwdAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t  cudnnTohipConvolutionFwdAlgo(cudnnConvolutionFwdAlgo_t  in, 
                                            hipdnnConvolutionFwdAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_GEMM;
        break;    
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        break;
    case CUDNN_CONVOLUTION_FWD_ALGO_COUNT :
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_COUNT;
        break;
    }
    return retVal;
} 

int ConvolutionFwdAlgoCount()
{
    return (int)CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
}

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t  GetConvolutionFwdAlgo(int i)
{
    hipdnnConvolutionFwdAlgo_t retVal;
    cudnnConvolutionFwdAlgo_t cualgo;
    
    if( i < ConvolutionFwdAlgoCount() )
    {
        cualgo = (cudnnConvolutionFwdAlgo_t)i;
    }
    else
    {
        //for protection
        cualgo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    }
    cudnnTohipConvolutionFwdAlgo(cualgo,&retVal);  //HGSOS does not compile, these functions need fwd declaration, or in hipdnn_cudnn header.
    
    return retVal;
}


//===========================================================


hipdnnStatus_t  hipTocudnnConvolutionBwdFilterAlgo(    hipdnnConvolutionBwdFilterAlgo_t in, 
                                                    cudnnConvolutionBwdFilterAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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
    }
    return retVal;
} 

hipdnnStatus_t  cudnnTohipConvolutionBwdFilterAlgo(    cudnnConvolutionBwdFilterAlgo_t  in, 
                                                    hipdnnConvolutionBwdFilterAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
        break;    
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
        break;
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT :
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
        break;
    }
    return retVal;
} 

int ConvolutionBwdFilterAlgoCount()
{
    return (int)CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
}
                                                
// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t  GetConvolutionBwdFilterAlgo(int i)
{
    hipdnnConvolutionBwdFilterAlgo_t retVal;
    cudnnConvolutionBwdFilterAlgo_t cualgo;
    
    if( i < ConvolutionBwdFilterAlgoCount() )
    {
        cualgo = (cudnnConvolutionBwdFilterAlgo_t)i;
    }
    else
    {
        //for protection
        cualgo = (cudnnConvolutionBwdFilterAlgo_t)CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
    }
    cudnnTohipConvolutionBwdFilterAlgo(cualgo,&retVal);
    
    return retVal;
}

//=============================================================================

hipdnnStatus_t  cudnnTohipConvolutionBwdDataAlgo(    cudnnConvolutionBwdDataAlgo_t in, 
                                                    hipdnnConvolutionBwdDataAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {    
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

hipdnnStatus_t  hipTocudnnConvolutionBwdDataAlgo(    hipdnnConvolutionBwdDataAlgo_t  in, 
                                                    cudnnConvolutionBwdDataAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {    
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0 :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1 :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
        break;    
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT :
        *out = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        break;
    }
    return retVal;
} 

int ConvolutionBwdDataAlgoCount()
{
    return (int)HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
}

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t  GetConvolutionBwdDataAlgo(int i)
{
    hipdnnConvolutionBwdDataAlgo_t retVal;
    cudnnConvolutionBwdDataAlgo_t cualgo;
    
    if( i < ConvolutionBwdDataAlgoCount() )
    {
        cualgo = (cudnnConvolutionBwdDataAlgo_t)i;
    }
    else
    {
        //for protection
        cualgo = (cudnnConvolutionBwdDataAlgo_t)CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
    }
    cudnnTohipConvolutionBwdDataAlgo(cualgo,&retVal);
    
    return retVal;
}
//=============================================================================

hipdnnStatus_t  hipTocudnnConvolutionBwdDataPreference(    hipdnnConvolutionBwdDataPreference_t  in, 
                                                    cudnnConvolutionBwdDataPreference_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {    
    case HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE :
        *out = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST :
        *out = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
        break;
    case HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT :
        *out = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        break;
    }
    return retVal;
}     

//==================================================

hipdnnStatus_t  hipTocudnnSoftmaxAlgorithm(    hipdnnSoftmaxAlgorithm_t in, 
                                            cudnnSoftmaxAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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


hipdnnStatus_t  hipTocudnnSoftmaxMode(    hipdnnSoftmaxMode_t in, 
                                        cudnnSoftmaxMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t  hipTocudnnTensorFormat(    hipdnnTensorFormat_t in, 
                                        cudnnTensorFormat_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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
} ;

//===========================================================

hipdnnStatus_t  hipTocudnnNanPropagation(    hipdnnNanPropagation_t in, 
                                            cudnnNanPropagation_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_NOT_PROPAGATE_NAN:
        *out = CUDNN_NOT_PROPAGATE_NAN;
        break;
    case HIPDNN_PROPAGATE_NAN:
        *out = CUDNN_PROPAGATE_NAN;
        break;
    }
    return retVal;
} 

hipdnnStatus_t  cudnnTohipNanPropagation(    cudnnNanPropagation_t in, 
                                            hipdnnNanPropagation_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t  hipTocudnnConvolutionFwdPreference(    hipdnnConvolutionFwdPreference_t in, 
                                            cudnnConvolutionFwdPreference_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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


hipdnnStatus_t  hipTocudnnConvolutionBwdFilterPreference(hipdnnConvolutionBwdFilterPreference_t in,                                                         cudnnConvolutionBwdFilterPreference_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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


//=============================================================================

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle)
{
    return cudnnTohipdnnStatus(cudnnCreate(handle));
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    return cudnnTohipdnnStatus(cudnnDestroy(handle));
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId)
{
    return cudnnTohipdnnStatus(cudnnSetStream(handle, streamId));
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipdnnStream_t *streamId)
{
    return cudnnTohipdnnStatus(cudnnGetStream(handle, streamId));
}

hipdnnStatus_t hipdnnCreateTensorDescriptor(hipdnnTensorDescriptor_t *tensorDesc)
{
    return cudnnTohipdnnStatus(cudnnCreateTensorDescriptor(tensorDesc));
}

//===============================

hipdnnStatus_t  hipdnnSetTensor4dDescriptor(    hipdnnTensorDescriptor_t tensorDesc,
                                            hipdnnTensorFormat_t format, 
                                            hipdnnDataType_t dataType,
                                            int n, int c, int h, int w)    
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnDataType_t cuDT;
    cudnnTensorFormat_t cuTF;
    
    retVal = hipTocudnnDataType(dataType, &cuDT);
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
 
     retVal = hipTocudnnTensorFormat(format, &cuTF);
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    return cudnnTohipdnnStatus(
                cudnnSetTensor4dDescriptor(    tensorDesc,
                                            cuTF, 
                                            cuDT,
                                            n, c, h, w));                
                
}            

        
//=============================================================================

hipdnnStatus_t hipdnnGetTensor4dDescriptor(    hipdnnTensorDescriptor_t tensorDesc,
                                            hipdnnDataType_t *dataType,
                                            int *n, int *c, int *h, int *w,
                                            int *nStride, int *cStride,
                                            int *hStride, int *wStride)
{
    cudnnDataType_t cudT;    
    hipdnnStatus_t    retVal;                                
                                            
    retVal = cudnnTohipdnnStatus(
                cudnnGetTensor4dDescriptor( tensorDesc,
                                            &cudT,
                                            n, c, h, w,
                                            nStride, cStride,
                                            hStride, wStride));
                            
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    return cudnnTohipDataType(cudT, dataType);

}


//=============================================================================

hipdnnStatus_t hipdnnDestroyTensorDescriptor(hipdnnTensorDescriptor_t tensorDesc)
{
return cudnnTohipdnnStatus(cudnnDestroyTensorDescriptor(tensorDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnAddTensor(    hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t     aDesc, 
                                const void *A,
                                const void *beta,
                                const hipdnnTensorDescriptor_t    cDesc, void *C)
{
    return cudnnTohipdnnStatus(
                cudnnAddTensor( handle,
                                alpha,
                                aDesc, 
                                A,
                                beta,
                                cDesc, 
                                C));
}

//======================HGSOS======================!

hipdnnStatus_t
hipdnnOpTensor(    hipdnnHandle_t handle,
                const hipdnnOpTensorDescriptor_t opTensorDesc,
                const void *alpha1,
                const hipdnnTensorDescriptor_t aDesc, 
                const void *A,
                const void *alpha2,
                const hipdnnTensorDescriptor_t bDesc, 
                const void *B,
                const void *beta,
                const hipdnnTensorDescriptor_t cDesc, 
                void *C)
{
                
    return cudnnTohipdnnStatus(
                cudnnOpTensor( handle,opTensorDesc, alpha1, 
                                aDesc,A,alpha2,bDesc, B,beta, cDesc, C));
}
//======

hipdnnStatus_t hipdnnSetTensor(    hipdnnHandle_t handle,
                                const hipdnnTensorDescriptor_t yDesc, 
                                void *y,
                                const void *valuePtr)
{
return cudnnTohipdnnStatus(cudnnSetTensor(handle, yDesc, y, valuePtr));
}

//==========

hipdnnStatus_t hipdnnScaleTensor(    hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t yDesc, 
                                    void *y,
                                    const void *alpha)
{
return cudnnTohipdnnStatus( cudnnScaleTensor(handle,yDesc,y,alpha));
}
//=============================================================================

hipdnnStatus_t
hipdnnCreateFilterDescriptor(hipdnnFilterDescriptor_t *filterDesc)
{
return cudnnTohipdnnStatus( 
            cudnnCreateFilterDescriptor(filterDesc));    
    
}

//=============================================================================


hipdnnStatus_t
hipdnnCreateConvolutionDescriptor(hipdnnConvolutionDescriptor_t *convDesc)
{
return cudnnTohipdnnStatus( cudnnCreateConvolutionDescriptor(convDesc));

}
//=====


hipdnnStatus_t
hipdnnSetConvolution2dDescriptor(    hipdnnConvolutionDescriptor_t convDesc,
                                    int pad_h, int pad_w,
                                    int u, int v,
                                    int upscalex, int upscaley,
                                    hipdnnConvolutionMode_t mode,
                                    hipdnnDataType_t computeType)  
{

    hipdnnStatus_t retVal;
    cudnnDataType_t cuDT;

    retVal = hipTocudnnDataType(computeType, &cuDT);
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    return cudnnTohipdnnStatus(
                cudnnSetConvolution2dDescriptor(convDesc, 
                                                pad_h, pad_w,
                                                u,  v,
                                                upscalex,  upscaley,
                                                hipTocudnnConvolutionMode(mode),
                                                cuDT));
}

//=============================================================================

hipdnnStatus_t
hipdnnGetConvolution2dDescriptor(const hipdnnConvolutionDescriptor_t convDesc,
                                int *pad_h, int *pad_y,
                                int *u, int *v,
                                int *upscalex, int *upscaley,
                                hipdnnConvolutionMode_t *mode,
                                hipdnnDataType_t *computeType)
{
    hipdnnStatus_t    retVal;        
    cudnnConvolutionMode_t cuMode;
    cudnnDataType_t cutype;

    retVal = cudnnTohipdnnStatus( 
                cudnnGetConvolution2dDescriptor(    convDesc,
                                                    pad_h, pad_y,
                                                    u, v,
                                                    upscalex, upscaley,
                                                    &cuMode,
                                                    &cutype));
    
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    *mode = cudnnTohipConvolutionMode(cuMode);
    
    return cudnnTohipDataType(cutype, computeType);
}
//===========

hipdnnStatus_t
hipdnnGetConvolution2dForwardOutputDim(
            const hipdnnConvolutionDescriptor_t convDesc, //HGSOSOS const
            const hipdnnTensorDescriptor_t inputTensorDesc,
            const hipdnnFilterDescriptor_t filterDesc,
            int *n, int *c, int *h, int *w)
{
return     cudnnTohipdnnStatus(
                    cudnnGetConvolution2dForwardOutputDim(
                    convDesc,
                    inputTensorDesc,
                    filterDesc,
                    n, c, h, w));
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyConvolutionDescriptor(hipdnnConvolutionDescriptor_t convDesc)
{
return     cudnnTohipdnnStatus(cudnnDestroyConvolutionDescriptor(convDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnFindConvolutionForwardAlgorithm(    hipdnnHandle_t handle,
                                        const hipdnnTensorDescriptor_t xDesc,
                                        const hipdnnFilterDescriptor_t wDesc,
                                        const hipdnnConvolutionDescriptor_t convDesc,
                                        const hipdnnTensorDescriptor_t yDesc,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        hipdnnConvolutionFwdAlgoPerf_t *perfResults)
{
    return     cudnnTohipdnnStatus(
                cudnnFindConvolutionForwardAlgorithm(     handle,
                                                        xDesc,
                                                        wDesc,
                                                        convDesc,
                                                        yDesc,
                                                        requestedAlgoCount,
                                                        returnedAlgoCount,
                                                        perfResults));    
}


hipdnnStatus_t
hipdnnGetConvolutionForwardAlgorithm(    hipdnnHandle_t handle,
                                        const hipdnnTensorDescriptor_t xDesc,
                                        const hipdnnFilterDescriptor_t wDesc,
                                        const hipdnnConvolutionDescriptor_t convDesc,
                                        const hipdnnTensorDescriptor_t yDesc,
                                        hipdnnConvolutionFwdPreference_t preference,
                                        size_t memoryLimitInBytes,
                                        hipdnnConvolutionFwdAlgo_t *algo)
{
    cudnnConvolutionFwdAlgo_t cualgo;
    cudnnConvolutionFwdPreference_t cupref;
    hipdnnStatus_t retVal;
    
    retVal =  hipTocudnnConvolutionFwdPreference(preference, &cupref);
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    retVal = cudnnTohipdnnStatus(
                cudnnGetConvolutionForwardAlgorithm( handle,
                                                    xDesc,
                                                    wDesc,
                                                    convDesc,
                                                    yDesc,
                                                    cupref,
                                                    memoryLimitInBytes,
                                                    &cualgo ));
                                    
    if( retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;
    
    return cudnnTohipConvolutionFwdAlgo(cualgo, algo);
    
}




hipdnnStatus_t
hipdnnFindConvolutionForwardAlgorithmEx(hipdnnHandle_t handle,
                                        const hipdnnTensorDescriptor_t xDesc,
                                        const void *x,
                                        const hipdnnFilterDescriptor_t wDesc,
                                        const void *w,
                                        const hipdnnConvolutionDescriptor_t convDesc,
                                        const hipdnnTensorDescriptor_t yDesc,
                                        void *y,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        hipdnnConvolutionFwdAlgoPerf_t *perfResults,
                                        void *workSpace,
                                        size_t workSpaceSizeInBytes)
{
                                    
return     cudnnTohipdnnStatus(
                cudnnFindConvolutionForwardAlgorithmEx( handle,
                                                        xDesc,
                                                        x,
                                                        wDesc,
                                                        w,
                                                        convDesc,
                                                        yDesc,
                                                        y,
                                                        requestedAlgoCount,
                                                        returnedAlgoCount,
                                                        perfResults,
                                                        workSpace,
                                                        workSpaceSizeInBytes));
}



//=============================================================================


hipdnnStatus_t
hipdnnGetConvolutionForwardWorkspaceSize(    hipdnnHandle_t handle,
                                            const hipdnnTensorDescriptor_t xDesc,  
                                            const hipdnnFilterDescriptor_t wDesc,  
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t yDesc,
                                            hipdnnConvolutionFwdAlgo_t algo,  
                                            size_t *sizeInBytes)
{
    cudnnConvolutionFwdAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionFwdAlgo(algo, &cualgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            xDesc,  
                                            wDesc,  
                                            convDesc,
                                            yDesc,
                                            cualgo,  
                                            sizeInBytes));

}

//=============================================================================

hipdnnStatus_t
hipdnnConvolutionForward(    hipdnnHandle_t handle,
                            const void *alpha,
                            const hipdnnTensorDescriptor_t xDesc,
                            const void *x,
                            const hipdnnFilterDescriptor_t wDesc,
                            const void *w,
                            const hipdnnConvolutionDescriptor_t convDesc,
                            hipdnnConvolutionFwdAlgo_t algo,
                            void *workSpace,
                            size_t workSpaceSizeInBytes,  
                            const void *beta,
                            const hipdnnTensorDescriptor_t yDesc,
                            void *y)
{
    cudnnConvolutionFwdAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionFwdAlgo(algo, &cualgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(
                    cudnnConvolutionForward(handle,
                                            alpha,
                                            xDesc,
                                            x,
                                            wDesc,
                                            w,
                                            convDesc,
                                            cualgo,
                                            workSpace,
                                            workSpaceSizeInBytes,  
                                            beta,
                                            yDesc,
                                            y));

}

//=============================================================================

hipdnnStatus_t
hipdnnConvolutionBackwardBias(    hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const void *beta,
                                const hipdnnTensorDescriptor_t dbDesc,
                                void *db)
{
    return     cudnnTohipdnnStatus(
                cudnnConvolutionBackwardBias(     handle,
                                                alpha,
                                                dyDesc,
                                                dy,
                                                beta,
                                                dbDesc,
                                                db));

}

//=============================================================================

hipdnnStatus_t
hipdnnFindConvolutionBackwardFilterAlgorithm(    hipdnnHandle_t handle,
                                                const hipdnnTensorDescriptor_t xDesc,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnFilterDescriptor_t dwDesc,
                                                const int requestedAlgoCount,
                                                int *returnedAlgoCount,
                                                hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
    
    return     cudnnTohipdnnStatus(
        cudnnFindConvolutionBackwardFilterAlgorithm( handle,
        xDesc,
        dyDesc,
        convDesc,
        dwDesc,
        requestedAlgoCount,
        returnedAlgoCount,
        perfResults));
}

hipdnnStatus_t
hipdnnGetConvolutionBackwardFilterAlgorithm(hipdnnHandle_t handle,
                                            const hipdnnTensorDescriptor_t xDesc,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnFilterDescriptor_t dwDesc,
                                            hipdnnConvolutionBwdFilterPreference_t preference,
                                            size_t memoryLimitInBytes,
                                            hipdnnConvolutionBwdFilterAlgo_t *algo)
{
    cudnnConvolutionBwdFilterPreference_t cupreference;
    cudnnConvolutionBwdFilterAlgo_t cualgo;
    
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    
    retVal =  hipTocudnnConvolutionBwdFilterPreference(    preference, 
                                                        &cupreference);
    if(retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return     cudnnTohipdnnStatus(
                cudnnGetConvolutionBackwardFilterAlgorithm( handle,
                                                            xDesc,
                                                            dyDesc,
                                                            convDesc,
                                                            dwDesc,
                                                            cupreference,
                                                            memoryLimitInBytes,
                                                            &cualgo));
                                                            
                                                            
}

hipdnnStatus_t
hipdnnFindConvolutionBackwardFilterAlgorithmEx(    hipdnnHandle_t handle,
                                                const hipdnnTensorDescriptor_t xDesc,
                                                const void *x,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const void *dy,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnFilterDescriptor_t dwDesc,
                                                void *dw,
                                                const int requestedAlgoCount,
                                                int *returnedAlgoCount,
                                                hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                                void *workSpace,
                                                size_t workSpaceSizeInBytes)
{
                                            
    return     cudnnTohipdnnStatus(
                cudnnFindConvolutionBackwardFilterAlgorithmEx(    handle,
                                                                xDesc,
                                                                x,
                                                                dyDesc,
                                                                dy,
                                                                convDesc,
                                                                dwDesc,
                                                                dw,
                                                                requestedAlgoCount,
                                                                returnedAlgoCount,
                                                                perfResults,
                                                                workSpace,
                                                                workSpaceSizeInBytes));

}

//=============================================================================


hipdnnStatus_t
hipdnnGetConvolutionBackwardFilterWorkspaceSize(hipdnnHandle_t handle,
                                                const hipdnnTensorDescriptor_t xDesc,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnFilterDescriptor_t dwDesc,  
                                                hipdnnConvolutionBwdFilterAlgo_t algo,
                                                size_t *sizeInBytes)
{
    cudnnConvolutionBwdFilterAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionBwdFilterAlgo(algo, &cualgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(
                cudnnGetConvolutionBackwardFilterWorkspaceSize( handle,
                                                                xDesc,
                                                                dyDesc,
                                                                convDesc,
                                                                dwDesc,  
                                                                cualgo,
                                                                sizeInBytes));
}


//=============================================================================

hipdnnStatus_t
hipdnnConvolutionBackwardFilter(    hipdnnHandle_t handle,
                                    const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const hipdnnTensorDescriptor_t dyDesc,
                                    const void *dy,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    hipdnnConvolutionBwdFilterAlgo_t algo,
                                    void *workSpace,
                                    size_t workSpaceSizeInBytes,
                                    const void *beta,
                                    const hipdnnFilterDescriptor_t dwDesc,
                                    void *dw)
{
    cudnnConvolutionBwdFilterAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionBwdFilterAlgo(algo, &cualgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(
                cudnnConvolutionBackwardFilter(    handle,
                                                alpha,
                                                xDesc,
                                                x,
                                                dyDesc,
                                                dy,
                                                convDesc,
                                                cualgo,
                                                workSpace,
                                                workSpaceSizeInBytes,
                                                beta,
                                                dwDesc,
                                                dw));
}

//=============================================================================

hipdnnStatus_t
hipdnnGetConvolutionBackwardDataWorkspaceSize(    hipdnnHandle_t handle,
                                                const hipdnnFilterDescriptor_t wDesc,
                                                const hipdnnTensorDescriptor_t dyDesc,
                                                const hipdnnConvolutionDescriptor_t convDesc,
                                                const hipdnnTensorDescriptor_t dxDesc,
                                                hipdnnConvolutionBwdDataAlgo_t algo,  
                                                size_t *sizeInBytes)
{
    cudnnConvolutionBwdDataAlgo_t cualgo;
    hipdnnStatus_t retVal;
    
    retVal = hipTocudnnConvolutionBwdDataAlgo( algo, &cualgo );

    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(
                cudnnGetConvolutionBackwardDataWorkspaceSize(     handle,
                                                                wDesc,
                                                                dyDesc,
                                                                convDesc,
                                                                dxDesc,
                                                                cualgo,  
                                                                sizeInBytes));
}


//=============================================================================

hipdnnStatus_t
hipdnnFindConvolutionBackwardDataAlgorithm(    hipdnnHandle_t handle,
                                            const hipdnnFilterDescriptor_t wDesc,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t dxDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            hipdnnConvolutionBwdDataAlgoPerf_t *perfResults)
{
    return     cudnnTohipdnnStatus(
                cudnnFindConvolutionBackwardDataAlgorithm(     handle,
                                                            wDesc,
                                                            dyDesc,
                                                            convDesc,
                                                            dxDesc,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount,
                                                            perfResults));
}

hipdnnStatus_t
hipdnnGetConvolutionBackwardDataAlgorithm(    hipdnnHandle_t handle,
                                            const hipdnnFilterDescriptor_t wDesc,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t dxDesc,
                                            hipdnnConvolutionBwdDataPreference_t preference,
                                            size_t memoryLimitInBytes,
                                            hipdnnConvolutionBwdDataAlgo_t *algo)
{
    cudnnConvolutionBwdDataPreference_t cupreference;
    cudnnConvolutionBwdDataAlgo_t cualgo;
    
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    
    retVal =  hipTocudnnConvolutionBwdDataPreference(    preference, 
                                                        &cupreference);
    if(retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    retVal = cudnnTohipdnnStatus(
                cudnnGetConvolutionBackwardDataAlgorithm(     handle,
                                                            wDesc,
                                                            dyDesc,
                                                            convDesc,
                                                            dxDesc,
                                                            cupreference,
                                                            memoryLimitInBytes,
                                                            &cualgo));
                                                            
    if(retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return  cudnnTohipConvolutionBwdDataAlgo(    cualgo, algo );
                                                    
}


hipdnnStatus_t
hipdnnFindConvolutionBackwardDataAlgorithmEx(hipdnnHandle_t handle,
                                            const hipdnnFilterDescriptor_t wDesc,
                                            const void *w,
                                            const hipdnnTensorDescriptor_t dyDesc,
                                            const void *dy,
                                            const hipdnnConvolutionDescriptor_t convDesc,
                                            const hipdnnTensorDescriptor_t dxDesc,
                                            void *dx,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            hipdnnConvolutionBwdDataAlgoPerf_t *perfResults,
                                            void *workSpace,
                                            size_t workSpaceSizeInBytes)
{
    return     cudnnTohipdnnStatus(
                cudnnFindConvolutionBackwardDataAlgorithmEx(handle,
                                                            wDesc,
                                                            w,
                                                            dyDesc,
                                                            dy,
                                                            convDesc,
                                                            dxDesc,
                                                            dx,
                                                            requestedAlgoCount,
                                                            returnedAlgoCount,
                                                            perfResults,
                                                            workSpace,
                                                            workSpaceSizeInBytes));
}



//=============================================================================


hipdnnStatus_t
hipdnnConvolutionBackwardData(     hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnFilterDescriptor_t wDesc,
                                const void *w,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnConvolutionDescriptor_t convDesc,
                                hipdnnConvolutionBwdDataAlgo_t algo,
                                void *workSpace,
                                size_t workSpaceSizeInBytes,
                                const void *beta,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx)
{
    cudnnConvolutionBwdDataAlgo_t cualgo;
    hipdnnStatus_t retVal = hipTocudnnConvolutionBwdDataAlgo(algo, &cualgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    
    return     cudnnTohipdnnStatus(
                cudnnConvolutionBackwardData(     handle,
                                                alpha,
                                                wDesc,
                                                w,
                                                dyDesc,
                                                dy,
                                                convDesc,
                                                cualgo,
                                                workSpace,
                                                workSpaceSizeInBytes,
                                                beta,
                                                dxDesc,
                                                dx));

}

//=============================================================================

hipdnnStatus_t
hipdnnSoftmaxForward(hipdnnHandle_t handle,
                    hipdnnSoftmaxAlgorithm_t algo,  
                    hipdnnSoftmaxMode_t mode,   
                    const void *alpha,
                    const hipdnnTensorDescriptor_t xDesc,
                    const void *x, 
                    const void *beta,
                    const hipdnnTensorDescriptor_t yDesc,
                    void *y)
{
    
    cudnnSoftmaxAlgorithm_t cuSMalgo;
    hipdnnStatus_t retVal = hipTocudnnSoftmaxAlgorithm(algo, &cuSMalgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    cudnnSoftmaxMode_t cuSMMode;
    retVal = hipTocudnnSoftmaxMode(mode, &cuSMMode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(
                cudnnSoftmaxForward(handle,
                                    cuSMalgo,  
                                    cuSMMode,   
                                    alpha,
                                    xDesc,
                                    x, 
                                    beta,
                                    yDesc,
                                    y));    
}

//=============================================================================

hipdnnStatus_t
hipdnnSoftmaxBackward(    hipdnnHandle_t handle,
                        hipdnnSoftmaxAlgorithm_t algo,  
                        hipdnnSoftmaxMode_t mode,
                        const void *alpha,
                        const hipdnnTensorDescriptor_t yDesc,
                        const void *y,
                        const hipdnnTensorDescriptor_t dyDesc,
                        const void *dy, const void *beta,
                        const hipdnnTensorDescriptor_t dxDesc,
                        void *dx)
{
    
    cudnnSoftmaxAlgorithm_t cuSMalgo;
    hipdnnStatus_t retVal = hipTocudnnSoftmaxAlgorithm(algo, &cuSMalgo);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    cudnnSoftmaxMode_t cuSMMode;
    retVal = hipTocudnnSoftmaxMode(mode, &cuSMMode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    
    return     cudnnTohipdnnStatus(
                cudnnSoftmaxBackward(    handle,
                                        cuSMalgo,  
                                        cuSMMode,
                                        alpha,
                                        yDesc,
                                        y,
                                        dyDesc,
                                        dy, beta,
                                        dxDesc,
                                        dx));
}



//=============================================================================

hipdnnStatus_t
hipdnnCreatePoolingDescriptor(hipdnnPoolingDescriptor_t *poolingDesc)
{
    return     cudnnTohipdnnStatus(
                cudnnCreatePoolingDescriptor(poolingDesc));
}
//=============================================================================

hipdnnStatus_t
hipdnnSetPooling2dDescriptor(    hipdnnPoolingDescriptor_t poolingDesc,
                                hipdnnPoolingMode_t mode,
                                hipdnnNanPropagation_t maxpoolingNanOpt, 
                                int windowHeight,
                                int windowWidth,
                                int verticalPadding,
                                int horizontalPadding,
                                int verticalStride,
                                int horizontalStride)
{

    cudnnPoolingMode_t cuPMode;
    hipdnnStatus_t retVal = hipTocudnnPoolingMode(mode, &cuPMode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    cudnnNanPropagation_t cuNaN;
    retVal = hipTocudnnNanPropagation(maxpoolingNanOpt, &cuNaN);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return     cudnnTohipdnnStatus(    
                cudnnSetPooling2dDescriptor(
                poolingDesc,
                cuPMode,
                cuNaN, 
                windowHeight,
                windowWidth,
                verticalPadding,
                horizontalPadding,
                verticalStride,
                horizontalStride));
    
}


//=============================================================================

hipdnnStatus_t
hipdnnGetPooling2dDescriptor(    const hipdnnPoolingDescriptor_t poolingDesc,
                                hipdnnPoolingMode_t *mode,
                                hipdnnNanPropagation_t *maxpoolingNanOpt, 
                                int *windowHeight,
                                int *windowWidth,
                                int *verticalPadding,
                                int *horizontalPadding,
                                int *verticalStride,
                                int *horizontalStride)
{
    hipdnnStatus_t retVal;
    cudnnPoolingMode_t cupmmode;
    cudnnNanPropagation_t cumaxpoolingNanOpt; 

    retVal = cudnnTohipdnnStatus(    
                cudnnGetPooling2dDescriptor( poolingDesc,
                                            &cupmmode,
                                            &cumaxpoolingNanOpt,
                                            windowHeight,
                                            windowWidth,
                                            verticalPadding,
                                            horizontalPadding,
                                            verticalStride,
                                            horizontalStride));
                            
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    retVal = cudnnTohipPoolingMode( cupmmode, mode );
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return cudnnTohipNanPropagation( cumaxpoolingNanOpt, maxpoolingNanOpt );

}

//=============================================================================

hipdnnStatus_t
hipdnnGetPooling2dForwardOutputDim(    const hipdnnPoolingDescriptor_t poolingDesc,
                                    const hipdnnTensorDescriptor_t inputTensorDesc,
                                    int *n, int *c, int *h, int *w)
{
    return cudnnTohipdnnStatus(    
                cudnnGetPooling2dForwardOutputDim(    poolingDesc,
                                                    inputTensorDesc,
                                                    n, c, h, w));
}

//=============================================================================

hipdnnStatus_t
hipdnnDestroyPoolingDescriptor(hipdnnPoolingDescriptor_t poolingDesc)
{
    return cudnnTohipdnnStatus(    cudnnDestroyPoolingDescriptor(poolingDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingForward(hipdnnHandle_t handle,
                                    const hipdnnPoolingDescriptor_t poolingDesc,
                                    const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const void *beta,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    void *y)
{
return cudnnTohipdnnStatus( 
            cudnnPoolingForward(handle,
                                poolingDesc,
                                alpha,
                                xDesc,
                                x,
                                beta,
                                yDesc,
                                y));
}


//=============================================================================

hipdnnStatus_t
hipdnnPoolingBackward(    hipdnnHandle_t handle,
                        const hipdnnPoolingDescriptor_t poolingDesc,
                        const void *alpha,
                        const hipdnnTensorDescriptor_t yDesc,
                        const void *y,
                        const hipdnnTensorDescriptor_t dyDesc,
                        const void *dy,
                        const hipdnnTensorDescriptor_t xDesc,
                        const void *x, const void *beta,
                        const hipdnnTensorDescriptor_t dxDesc,
                        void *dx)
{
return cudnnTohipdnnStatus( 
            cudnnPoolingBackward(    handle,
                                    poolingDesc,
                                    alpha,
                                    yDesc,
                                    y,
                                    dyDesc,
                                    dy,
                                    xDesc,
                                    x, 
                                    beta,
                                    dxDesc,
                                    dx));
}

//=============================================================================

hipdnnStatus_t
hipdnnCreateActivationDescriptor(hipdnnActivationDescriptor_t *activationDesc)
{
    return cudnnTohipdnnStatus(cudnnCreateActivationDescriptor(activationDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnSetActivationDescriptor(    hipdnnActivationDescriptor_t activationDesc, //HGSOS const
                                hipdnnActivationMode_t mode,
                                hipdnnNanPropagation_t reluNanOpt, 
                                double reluCeiling)
{
    cudnnActivationMode_t cuAMode;
    hipdnnStatus_t retVal;
    cudnnNanPropagation_t cuNaN;
    
    retVal = hipTocudnnActivationMode(mode, &cuAMode);
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    retVal = hipTocudnnNanPropagation(reluNanOpt, &cuNaN);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return cudnnTohipdnnStatus(
            cudnnSetActivationDescriptor(
                                activationDesc, //const
                                cuAMode,
                                cuNaN, 
                                reluCeiling));

}


//=============================================================================

hipdnnStatus_t
hipdnnGetActivationDescriptor(    const hipdnnActivationDescriptor_t activationDesc,
                                hipdnnActivationMode_t *mode,
                                hipdnnNanPropagation_t *reluNanOpt,  
                                double* reluCeiling)
{
    hipdnnStatus_t retVal;
    cudnnActivationMode_t cuactmode;
    cudnnNanPropagation_t cureluNanOpt;
    
    retVal = cudnnTohipdnnStatus(
            cudnnGetActivationDescriptor(    activationDesc,
                                            &cuactmode,
                                            &cureluNanOpt,  
                                            reluCeiling));

    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    retVal = cudnnTohipActivationMode( cuactmode, mode );
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return cudnnTohipNanPropagation( cureluNanOpt, reluNanOpt );
}


//=============================================================================

hipdnnStatus_t
hipdnnDestroyActivationDescriptor(hipdnnActivationDescriptor_t activationDesc)
{
    return     cudnnTohipdnnStatus(cudnnDestroyActivationDescriptor(activationDesc));
}
//=================

hipdnnStatus_t
hipdnnActivationForward(hipdnnHandle_t handle,
                        hipdnnActivationDescriptor_t activationDesc, //HGSOS not const in cudnn.
                        const void *alpha,
                        const hipdnnTensorDescriptor_t xDesc,
                        const void *x,
                        const void *beta,
                        const hipdnnTensorDescriptor_t yDesc,
                        void *y)
{
    return     cudnnTohipdnnStatus(
                cudnnActivationForward(    handle,
                                        activationDesc,
                                        alpha,
                                        xDesc,
                                        x,
                                        beta,
                                        yDesc,
                                        y));
}
//======================

hipdnnStatus_t
hipdnnActivationBackward(    hipdnnHandle_t handle,
                            hipdnnActivationDescriptor_t activationDesc,  //HGSOS not const in cuda
                            const void *alpha,
                            const hipdnnTensorDescriptor_t yDesc,
                            const void *y,
                            const hipdnnTensorDescriptor_t dyDesc,
                            const void *dy,
                            const hipdnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const hipdnnTensorDescriptor_t dxDesc,
                            void *dx)
{
    return     cudnnTohipdnnStatus(
                cudnnActivationBackward(     handle,
                                            activationDesc,
                                            alpha,
                                            yDesc,
                                            y,
                                            dyDesc,
                                            dy,
                                            xDesc,
                                            x, beta,
                                            dxDesc,
                                            dx));
}
//=============================================================================


hipdnnStatus_t
hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc)
{
    return     cudnnTohipdnnStatus(cudnnCreateLRNDescriptor(normDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnSetLRNDescriptor(    hipdnnLRNDescriptor_t normDesc,
                        hipdnnLRNMode_t mode,  
                        unsigned lrnN, double lrnAlpha,
                        double lrnBeta, double lrnK)
{
    
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnLRNMode_t cumode;
    
    retVal = hipTocudnnLRNMode(    mode, &cumode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return     cudnnTohipdnnStatus(
                cudnnSetLRNDescriptor(    normDesc,
                                        lrnN, lrnAlpha,
                                        lrnBeta, lrnK));

}

//================

hipdnnStatus_t
hipdnnGetLRNDescriptor(    hipdnnLRNDescriptor_t normDesc,
                        hipdnnLRNMode_t *mode, 
                        unsigned* lrnN, 
                        double* lrnAlpha,
                        double* lrnBeta, 
                        double* lrnK)
{
    
    *mode = HIPDNN_LRN_CROSS_CHANNEL;
  
    return cudnnTohipdnnStatus(
            cudnnGetLRNDescriptor(    normDesc,
                                    lrnN, 
                                    lrnAlpha,
                                    lrnBeta, 
                                    lrnK));
    
}


//=============================================================================

hipdnnStatus_t
hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t lrnDesc)
{
    return cudnnTohipdnnStatus(cudnnDestroyLRNDescriptor(lrnDesc));
}

//=============================================================================

hipdnnStatus_t
hipdnnLRNCrossChannelForward(    hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,  
                                const void* alpha,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const void *beta,
                                const hipdnnTensorDescriptor_t yDesc,
                                void *y)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnLRNMode_t cumode;
    
    retVal = hipTocudnnLRNMode(    lrnMode, &cumode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return cudnnTohipdnnStatus(
                cudnnLRNCrossChannelForward(handle,
                                            normDesc,
                                            cumode,  
                                            alpha,
                                            xDesc,
                                            x,
                                            beta,
                                            yDesc,
                                            y));
}

hipdnnStatus_t
hipdnnLRNCrossChannelForwardEx(    hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,  
                                const void* alpha,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const void *beta,
                                const hipdnnTensorDescriptor_t yDesc,
                                void *y,
                                size_t workspacesize,
                                void *workspace)
{
    return hipdnnLRNCrossChannelForward(handle,
                                        normDesc,
                                        lrnMode,  
                                        alpha,
                                        xDesc,
                                        x,
                                        beta,
                                        yDesc,
                                        y);
}


//=============================================================================

hipdnnStatus_t
hipdnnLRNCrossChannelBackward(    hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,  
                                const void* alpha,
                                const hipdnnTensorDescriptor_t yDesc,
                                const void *y,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x, const void *beta,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx)
{

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    cudnnLRNMode_t cumode;
    
    retVal = hipTocudnnLRNMode(    lrnMode, &cumode);
    
    if( retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;
    
    return cudnnTohipdnnStatus(
                cudnnLRNCrossChannelBackward(     handle,
                                                normDesc,
                                                cumode,  
                                                alpha,
                                                yDesc,
                                                y,
                                                dyDesc,
                                                dy,
                                                xDesc,
                                                x, beta,
                                                dxDesc,
                                                dx));
}

hipdnnStatus_t
hipdnnLRNCrossChannelBackwardEx(hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,  
                                const void* alpha,
                                const hipdnnTensorDescriptor_t yDesc,
                                const void *y,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x, const void *beta,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx,
                                size_t workspacesize,
                                void* workspace)
{
    return hipdnnLRNCrossChannelBackward(     handle,
                                            normDesc,
                                            lrnMode,  
                                            alpha,
                                            yDesc,
                                            y,
                                            dyDesc,
                                            dy,
                                            xDesc,
                                            x, beta,
                                            dxDesc,
                                            dx);    
}

//=============================================================================


hipdnnStatus_t
hipdnnDeriveBNTensorDescriptor(    hipdnnTensorDescriptor_t derivedBnDesc,
                                const hipdnnTensorDescriptor_t xDesc,
                                hipdnnBatchNormMode_t mode)
{
    return cudnnTohipdnnStatus(
                cudnnDeriveBNTensorDescriptor(     derivedBnDesc,
                                                xDesc,
                                                hipTocudnnBatchNormMode(mode)));
}

//=============================================================================

hipdnnStatus_t
hipdnnBatchNormalizationForwardTraining(    hipdnnHandle_t handle,
                                            hipdnnBatchNormMode_t mode,
                                            void *alpha, 
                                            void *beta,
                                            const hipdnnTensorDescriptor_t xDesc,
                                            const void *x,
                                            const hipdnnTensorDescriptor_t yDesc,
                                            void *y,
                                            const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                            void *bnScale, void *bnBias,
                                            double exponentialAverageFactor,
                                            void *resultRunningMean,
                                            void *resultRunningVariance,
                                            double epsilon,
                                            void *resultSaveMean,
                                            void *resultSaveInvVariance)
{
    return cudnnTohipdnnStatus(
                cudnnBatchNormalizationForwardTraining(    
                                        handle,
                                        hipTocudnnBatchNormMode(mode),
                                        alpha, 
                                        beta,
                                        xDesc,
                                        x,
                                        yDesc,
                                        y,
                                        bnScaleBiasMeanVarDesc,
                                        bnScale, bnBias,
                                        exponentialAverageFactor,
                                        resultRunningMean,
                                        resultRunningVariance,
                                        epsilon,
                                        resultSaveMean,
                                        resultSaveInvVariance));

}
//=============================================================================

hipdnnStatus_t
hipdnnnBatchNormalizationForwardInference(    hipdnnHandle_t handle,
                                            hipdnnBatchNormMode_t mode,
                                            void *alpha, 
                                            void *beta,
                                            const hipdnnTensorDescriptor_t xDesc,
                                            const void *x,
                                            const hipdnnTensorDescriptor_t yDesc,
                                            void *y,
                                            const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                            const void *bnScale, 
                                            const void *bnBias,
                                            const void *estimatedMean,
                                            const void *estimatedVariance,
                                            double epsilon)
{
    return cudnnTohipdnnStatus(
            cudnnBatchNormalizationForwardInference(     
                                            handle,
                                            hipTocudnnBatchNormMode(mode),
                                            alpha, 
                                            beta,
                                            xDesc,
                                            x,
                                            yDesc,
                                            y,
                                            bnScaleBiasMeanVarDesc,
                                            bnScale, bnBias,
                                            estimatedMean,
                                            estimatedVariance,
                                            epsilon));

}
//=============================================================================

hipdnnStatus_t
hipdnnBatchNormalizationBackward(hipdnnHandle_t handle,
                                hipdnnBatchNormMode_t mode,
                                const void *alphaDataDiff,
                                const void *betaDataDiff,
                                const void *alphaParamDiff,
                                const void *betaParamDiff,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnTensorDescriptor_t dxDesc,
                                void *dx,
                                const hipdnnTensorDescriptor_t bnScaleBiasDiffDesc,
                                const void *bnScale,
                                void *resultBnScaleDiff,
                                void *resultBnBiasDiff,
                                double epsilon,
                                const void *savedMean,
                                const void *savedInvVariance)
{
return cudnnTohipdnnStatus(
cudnnBatchNormalizationBackward( handle,
                                hipTocudnnBatchNormMode(mode),
                                alphaDataDiff,
                                betaDataDiff,
                                alphaParamDiff,
                                betaParamDiff,
                                xDesc,
                                x,
                                dyDesc,
                                dy,
                                dxDesc,
                                dx,
                                bnScaleBiasDiffDesc,
                                bnScale,
                                resultBnScaleDiff,
                                resultBnBiasDiff,
                                epsilon,
                                savedMean,
                                savedInvVariance));
}

