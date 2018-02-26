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
#include "iostream"


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

cudnnStatus_t hipdnnTocudnnStatus(hipdnnStatus_t  cStatus)
{
    cudnnStatus_t retVal;
    switch(cStatus)
    {
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

hipdnnStatus_t hipTocudnnMathType(hipdnnMathType_t in, cudnnMathType_t *out)
{
	switch(in)
	{
	case HIPDNN_DEFAULT_MATH:
	    *out = CUDNN_DEFAULT_MATH;
	    break;
	case HIPDNN_TENSOR_OP_MATH:
	    *out = CUDNN_TENSOR_OP_MATH;
	}
	return HIPDNN_STATUS_SUCCESS;
}


hipdnnStatus_t cudnnTohipMathType(cudnnMathType_t in, hipdnnMathType_t *out)
{
	switch(in)
	{
	case CUDNN_DEFAULT_MATH:
	    *out = HIPDNN_DEFAULT_MATH;
	    break;
	case CUDNN_TENSOR_OP_MATH:
	    *out = HIPDNN_TENSOR_OP_MATH;
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
    else if( in == HIPDNN_BATCHNORM_SPATIAL_PERSISTENT )
        return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    
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
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
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
}

hipdnnStatus_t  cudnnTohipTensorFormat(    cudnnTensorFormat_t in,
                                        hipdnnTensorFormat_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

//RNN Type Conversions

hipdnnStatus_t hipTocudnnRNNMode(hipdnnRNNMode_t in, cudnnRNNMode_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t cudnnTohipRNNMode(cudnnRNNMode_t in, hipdnnRNNMode_t *out)
{   
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t hipTocudnnDirectionMode(hipdnnDirectionMode_t in, cudnnDirectionMode_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_UNIDIRECTIONAL:
        *out = CUDNN_UNIDIRECTIONAL;
        break;
    case HIPDNN_BIDIRECTIONAL:
        *out = CUDNN_BIDIRECTIONAL;
        break;
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipDirectionMode(cudnnDirectionMode_t in, hipdnnDirectionMode_t *out)
{ 
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t hipTocudnnRNNInputMode(hipdnnRNNInputMode_t in, cudnnRNNInputMode_t *out) 
{ 
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
    case HIPDNN_LINEAR_INPUT:
        *out = CUDNN_LINEAR_INPUT;                                        
        break;
    case HIPDNN_SKIP_INPUT:                                              
        *out = CUDNN_SKIP_INPUT;                                         
        break;                                                              
    }
    return retVal;
}

hipdnnStatus_t cudnnTohipdnnRNNInputMode(cudnnRNNInputMode_t in, hipdnnRNNInputMode_t *out) 
{                                                                           
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;                          
    switch(in)                                                              
    {
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

hipdnnStatus_t hipTocudnnRNNAlgo(hipdnnRNNAlgo_t in, cudnnRNNAlgo_t *out) 
{ 
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t cudnnTohipdnnRNNAlgo(cudnnRNNAlgo_t in, hipdnnRNNAlgo_t *out)   
{  
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t cudnnTohipReduceTensorOp(cudnnReduceTensorOp_t in, hipdnnReduceTensorOp_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t hipTocudnnReduceTensorOp(hipdnnReduceTensorOp_t in, cudnnReduceTensorOp_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t cudnnTohipReduceTensorIndices(cudnnReduceTensorIndices_t in, hipdnnReduceTensorIndices_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t hipTocudnnReduceTensorIndices(hipdnnReduceTensorIndices_t in, cudnnReduceTensorIndices_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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


hipdnnStatus_t cudnnTohipIndicesType(cudnnIndicesType_t in, hipdnnIndicesType_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

hipdnnStatus_t hipTocudnnIndicesType(hipdnnIndicesType_t in, cudnnIndicesType_t *out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch(in)
    {
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

size_t hipdnnGetVersion()
{
    return cudnnGetVersion();
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
hipdnnSetConvolutionMathType(hipdnnConvolutionDescriptor_t convDesc,
                                                       hipdnnMathType_t mathType )
{

	hipdnnStatus_t retVal;
	cudnnMathType_t cuMT;

	retVal = hipTocudnnMathType(mathType, &cuMT);
	if(retVal == HIPDNN_STATUS_SUCCESS)
		return cudnnTohipdnnStatus(cudnnSetConvolutionMathType(convDesc, cuMT));
	return retVal;
}


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

    retVal = cudnnTohipdnnStatus(
                cudnnGetConvolutionBackwardFilterAlgorithm( handle,
                                                            xDesc,
                                                            dyDesc,
                                                            convDesc,
                                                            dwDesc,
                                                            cupreference,
                                                            memoryLimitInBytes,
                                                            &cualgo));

    if(retVal != HIPDNN_STATUS_SUCCESS )
        return retVal;

    return cudnnTohipConvolutionBwdFilterAlgo(cualgo, algo);
                                                            
                                                            
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
   /* cudnnConvolutionBwdFilterAlgo_t cualgo;
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
                                                dw));*/

	cudnnConvolutionBwdFilterAlgo_t cualgo;
        hipdnnStatus_t hstatus = hipTocudnnConvolutionBwdFilterAlgo(algo, &cualgo);
        if( hstatus != HIPDNN_STATUS_SUCCESS )
                return hstatus;
    std::cout<<"\n Done with algo change bro";
	//return cudnnTohipdnnStatus(cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw));
	return cudnnTohipdnnStatus(cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,cualgo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw));
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

hipdnnStatus_t hipdnnSetTensorNdDescriptor(
                                hipdnnTensorDescriptor_t             tensorDesc,
                                hipdnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[],
                                const int                           strideA[] )
{

	cudnnDataType_t cuDT;
	hipdnnStatus_t retval;
	retval = hipTocudnnDataType(dataType, &cuDT);
	if ( retval != HIPDNN_STATUS_SUCCESS)
		return retval;
	
	return cudnnTohipdnnStatus(
	  cudnnSetTensorNdDescriptor(
				tensorDesc,
				cuDT,
				nbDims,
				dimA,
				strideA));
}

hipdnnStatus_t hipdnnGetTensorNdDescriptor(
                                const hipdnnTensorDescriptor_t       tensorDesc,
                                int                                 nbDimsRequested,
                                hipdnnDataType_t                    *dataType,
                                int                                *nbDims,
                                int                                 dimA[],
                                int                                 strideA[] )
{
	cudnnDataType_t cuDT;
	hipdnnStatus_t retval;
	retval = cudnnTohipdnnStatus(
	  cudnnGetTensorNdDescriptor(
				tensorDesc,
				nbDimsRequested,
				&cuDT,
				nbDims,
				dimA,
				strideA));
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;
	return cudnnTohipDataType(cuDT, dataType);
}

hipdnnStatus_t hipdnnCreateDropoutDescriptor(hipdnnDropoutDescriptor_t * dropoutDesc)
{
        return cudnnTohipdnnStatus(
		cudnnCreateDropoutDescriptor(dropoutDesc));
}

hipdnnStatus_t hipdnnDropoutGetStatesSize(hipdnnHandle_t handle, size_t * sizeInBytes)
{
        return cudnnTohipdnnStatus(
		cudnnDropoutGetStatesSize(handle, sizeInBytes));
}

hipdnnStatus_t hipdnnSetDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc,
                                                    hipdnnHandle_t handle,
                                                    float dropout,
                                                    void * states,
                                                    size_t stateSizeInBytes,
                                                    unsigned long long seed)
{
        return cudnnTohipdnnStatus(
		cudnnSetDropoutDescriptor(dropoutDesc,
					  handle,
					  dropout,
					  states,
					  stateSizeInBytes,
					  seed));
}

hipdnnStatus_t hipdnnDestroyDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc)
{
	return cudnnTohipdnnStatus(
		cudnnDestroyDropoutDescriptor(dropoutDesc));
}

hipdnnStatus_t hipdnnSetFilterNdDescriptor(
                                hipdnnFilterDescriptor_t             filterDesc,
                                hipdnnDataType_t                     dataType, // image data type
                                hipdnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] )
{
	cudnnDataType_t cuDT;
	cudnnTensorFormat_t cuTF;
	hipdnnStatus_t retval;
	retval = hipTocudnnDataType(dataType, &cuDT);
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;
	
	retval = hipTocudnnTensorFormat(format, &cuTF);
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;
 	
        return cudnnTohipdnnStatus(
		cudnnSetFilterNdDescriptor(filterDesc,
					   cuDT,
					   cuTF,
					   nbDims,
					   filterDimA));
}

hipdnnStatus_t hipdnnGetFilterNdDescriptor(
                                const hipdnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                hipdnnDataType_t                    *dataType, // image data type
                                hipdnnTensorFormat_t                *format,
                                int                                *nbDims,
                                int                                 filterDimA[] )
{
	cudnnDataType_t cuDT;
        cudnnTensorFormat_t cuTF;
        hipdnnStatus_t retval;
	retval = cudnnTohipdnnStatus(
			cudnnGetFilterNdDescriptor(filterDesc,
						   nbDimsRequested,
						   &cuDT,
						   &cuTF,
						   nbDims,
						   filterDimA));
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;

	retval = cudnnTohipTensorFormat(cuTF, format);
	if (retval != HIPDNN_STATUS_SUCCESS)
                return retval;
	
        return cudnnTohipDataType(cuDT, dataType);
}


hipdnnStatus_t hipdnnDestroyFilterDescriptor(
                                hipdnnFilterDescriptor_t filterDesc )
{
        return cudnnTohipdnnStatus(
		cudnnDestroyFilterDescriptor(filterDesc));
}

hipdnnStatus_t hipdnnSetConvolutionNdDescriptor(
					hipdnnConvolutionDescriptor_t convDesc,
					int arrayLength, /* nbDims-2 size */
					const int padA[],
					const int filterStrideA[],
					const int dilationA[],
					hipdnnConvolutionMode_t mode,
					hipdnnDataType_t computeType ) // convolution data type
{
	cudnnDataType_t cuDT;
	cudnnConvolutionMode_t cuCM;
	hipdnnStatus_t retval;
	
	cuCM = hipTocudnnConvolutionMode(mode);
	
	retval = hipTocudnnDataType(computeType, &cuDT);
	if (retval != HIPDNN_STATUS_SUCCESS)
                return retval;

        return cudnnTohipdnnStatus(
		cudnnSetConvolutionNdDescriptor(convDesc,
						arrayLength,
						padA,
						filterStrideA,
						dilationA,
						cuCM,
						cuDT));
}

hipdnnStatus_t hipdnnSetPoolingNdDescriptor(hipdnnPoolingDescriptor_t poolingDesc,
					    const hipdnnPoolingMode_t mode,
					    const hipdnnNanPropagation_t maxpoolingNanOpt,
					    int nbDims,
					    const int windowDimA[],
					    const int paddingA[],
					    const int strideA[] )
{
	cudnnPoolingMode_t cuPM;
	cudnnNanPropagation_t cuNP;
	hipdnnStatus_t retval;

	retval = hipTocudnnPoolingMode(mode, &cuPM);
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;

	retval = hipTocudnnNanPropagation(maxpoolingNanOpt, &cuNP);
	if (retval != HIPDNN_STATUS_SUCCESS)
                return retval;

        return cudnnTohipdnnStatus(
		cudnnSetPoolingNdDescriptor(poolingDesc,
					    cuPM,
					    cuNP,
					    nbDims,
					    windowDimA,
					    paddingA,
					    strideA));
}

const char * hipdnnGetErrorString(hipdnnStatus_t status)
{
	cudnnStatus_t cstatus;
	cstatus = hipdnnTocudnnStatus(status);
	return cudnnGetErrorString(cstatus);
}

//RNN APIs

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t * rnnDesc)
{
        return cudnnTohipdnnStatus(
		cudnnCreateRNNDescriptor(rnnDesc));
}

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc)
{
        return cudnnTohipdnnStatus(
		cudnnDestroyRNNDescriptor(rnnDesc));
}

hipdnnStatus_t  hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                             const int minibatch,
                                             const hipdnnDataType_t dataType,
                                             hipdnnPersistentRNNPlan_t * plan)
{
	cudnnDataType_t cuDT;
	hipdnnStatus_t retval;
	
	retval = hipTocudnnDataType(dataType, &cuDT);
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;

        return cudnnTohipdnnStatus(
		cudnnCreatePersistentRNNPlan(rnnDesc,
					     minibatch,
					     cuDT,
					     plan));
					     
}
                                             
hipdnnStatus_t  hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                          hipdnnPersistentRNNPlan_t plan)
{
        return cudnnTohipdnnStatus(
		cudnnSetPersistentRNNPlan(rnnDesc, plan));
}
                                          
hipdnnStatus_t  hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan)
{
        return cudnnTohipdnnStatus(
		cudnnDestroyPersistentRNNPlan(plan));
}
                                          
hipdnnStatus_t  hipdnnSetRNNDescriptor_v6(hipdnnHandle_t handle, 
                                                hipdnnRNNDescriptor_t rnnDesc,
                                                const int hiddenSize, 
                                                const int numLayers, 
                                                hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                                hipdnnRNNInputMode_t inputMode,                                                 
                                                hipdnnDirectionMode_t direction, 
                                                hipdnnRNNMode_t mode, 
                                                hipdnnRNNAlgo_t algo, 
                                                hipdnnDataType_t dataType)
{
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
	
        return cudnnTohipdnnStatus(
		cudnnSetRNNDescriptor_v6(handle,
                                         rnnDesc,
                                         hiddenSize,
                                         numLayers,
                                         dropoutDesc,
                                         cuRIM,
                                         cuDM,
                                         cuRM,
                                         cuRA,
                                         cuDT));
}


hipdnnStatus_t  hipdnnSetRNNDescriptor(hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc,
                                                int hiddenSize, 
                                                int numLayers, 
                                                hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                                hipdnnRNNInputMode_t inputMode,                                                 
                                                hipdnnDirectionMode_t direction, 
                                                hipdnnRNNMode_t mode, 
						hipdnnRNNAlgo_t algo,
                                                hipdnnDataType_t dataType)
{
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

        return cudnnTohipdnnStatus(
		cudnnSetRNNDescriptor(handle, rnnDesc,
                                      hiddenSize,
                                      numLayers,
                                      dropoutDesc,
                                      cuRIM,
                                      cuDM,
                                      cuRM,
				                      cuRA,
                                      cuDT));
}



hipdnnStatus_t hipdnnSetRNNDescriptor_v5(hipdnnRNNDescriptor_t rnnDesc,
                                                    int hiddenSize,
                                                    int numLayers,
                                                    hipdnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
                                                    hipdnnRNNInputMode_t inputMode,
                                                    hipdnnDirectionMode_t direction,
                                                    hipdnnRNNMode_t mode,
                                                    hipdnnDataType_t dataType)
{
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
        return cudnnTohipdnnStatus(
                cudnnSetRNNDescriptor_v5(rnnDesc,
                                         hiddenSize,
                                         numLayers, 
                                         dropoutDesc,
                                         cuRIM,
                                         cuDM,
                                         cuRM,
                                         cuDT));
}



hipdnnStatus_t  hipdnnGetRNNWorkspaceSize( hipdnnHandle_t              handle,
                                                    const hipdnnRNNDescriptor_t rnnDesc,  
                                                    const int seqLength, 
                                                    const hipdnnTensorDescriptor_t    *xDesc,
                                                    size_t                     *sizeInBytes
                                                    )
{
        return cudnnTohipdnnStatus(
		cudnnGetRNNWorkspaceSize(handle,
					 rnnDesc,
					 seqLength,
					 xDesc,
					 sizeInBytes));
}
                                                      
hipdnnStatus_t  hipdnnGetRNNTrainingReserveSize( hipdnnHandle_t              handle,
                                                          const hipdnnRNNDescriptor_t rnnDesc,  
                                                          const int seqLength, 
                                                          const hipdnnTensorDescriptor_t    *xDesc,
                                                          size_t                     *sizeInBytes
                                                    )
{
        return cudnnTohipdnnStatus(
		cudnnGetRNNTrainingReserveSize(handle,
					       rnnDesc,
					       seqLength,
					       xDesc,
					       sizeInBytes));
}

                                                    
hipdnnStatus_t  hipdnnGetRNNParamsSize( hipdnnHandle_t              handle,
                                                 const hipdnnRNNDescriptor_t rnnDesc,  
                                                 const hipdnnTensorDescriptor_t    xDesc,                                                    
                                                 size_t                     *sizeInBytes,
                                                 hipdnnDataType_t dataType
                                                    )
{
	cudnnDataType_t cuDT;
	hipdnnStatus_t retval;

	retval = hipTocudnnDataType(dataType, &cuDT);
	if (retval != HIPDNN_STATUS_SUCCESS)
		return retval;

        return cudnnTohipdnnStatus(
		cudnnGetRNNParamsSize(handle,
				      rnnDesc,
				      xDesc,
				      sizeInBytes,
				      cuDT));
}

hipdnnStatus_t  hipdnnGetRNNLinLayerMatrixParams( hipdnnHandle_t              handle,
                             const hipdnnRNNDescriptor_t rnnDesc,  
                             const int layer,
                             const hipdnnTensorDescriptor_t xDesc, 
                             const hipdnnFilterDescriptor_t wDesc, 
                             const void * w, 
                             const int linLayerID,  
                             hipdnnFilterDescriptor_t linLayerMatDesc, 
                             void ** linLayerMat
                             )
{
        return cudnnTohipdnnStatus(
		cudnnGetRNNLinLayerMatrixParams(handle,
						rnnDesc,
						layer,
						xDesc,
						wDesc,
						w,
						linLayerID,
						linLayerMatDesc,
						linLayerMat));
}

hipdnnStatus_t  hipdnnGetRNNLinLayerBiasParams( hipdnnHandle_t              handle,
                             const hipdnnRNNDescriptor_t rnnDesc,  
                             const int layer,
                             const hipdnnTensorDescriptor_t xDesc, 
                             const hipdnnFilterDescriptor_t wDesc, 
                             const void * w, 
                             const int linLayerID, 
                             hipdnnFilterDescriptor_t linLayerBiasDesc, 
                             void ** linLayerBias                       
                             )
{
        return cudnnTohipdnnStatus(
		cudnnGetRNNLinLayerBiasParams(handle,
					      rnnDesc,
					      layer,
					      xDesc,
					      wDesc,
					      w,
					      linLayerID,
					      linLayerBiasDesc,
					      linLayerBias));
}

hipdnnStatus_t  hipdnnRNNForwardInference( hipdnnHandle_t handle, 
                                                    const hipdnnRNNDescriptor_t rnnDesc, 
                                                    const int seqLength, 
                                                    const hipdnnTensorDescriptor_t * xDesc, 
                                                    const void * x, 
                                                    const hipdnnTensorDescriptor_t hxDesc, 
                                                    const void * hx, 
                                                    const hipdnnTensorDescriptor_t cxDesc, 
                                                    const void * cx, 
                                                    const hipdnnFilterDescriptor_t wDesc, 
                                                    const void * w, 
                                                    const hipdnnTensorDescriptor_t *yDesc,  
                                                    void * y, 
                                                    const hipdnnTensorDescriptor_t hyDesc, 
                                                    void * hy, 
                                                    const hipdnnTensorDescriptor_t cyDesc, 
                                                    void * cy, 
                                                    void * workspace, 
                                                    size_t workSpaceSizeInBytes)
{
        return cudnnTohipdnnStatus(
		cudnnRNNForwardInference(handle,
                                          rnnDesc,
                                          seqLength,
                                          xDesc,
                                          x,
                                          hxDesc,
                                          hx,
                                          cxDesc,
                                          cx,
                                          wDesc,
                                          w,
                                          yDesc,
                                          y,
                                          hyDesc,
                                          hy,
                                          cyDesc,
                                          cy,
                                          workspace,
                                          workSpaceSizeInBytes));
}

hipdnnStatus_t  hipdnnRNNForwardTraining( hipdnnHandle_t handle, 
                                                   const hipdnnRNNDescriptor_t rnnDesc, 
                                                   const int seqLength, 
                                                   const hipdnnTensorDescriptor_t *xDesc, 
                                                   const void * x, 
                                                   const hipdnnTensorDescriptor_t hxDesc, 
                                                   const void * hx, 
                                                   const hipdnnTensorDescriptor_t cxDesc, 
                                                   const void * cx, 
                                                   const hipdnnFilterDescriptor_t wDesc, 
                                                   const void * w, 
                                                   const hipdnnTensorDescriptor_t *yDesc,  
                                                   void * y, 
                                                   const hipdnnTensorDescriptor_t hyDesc, 
                                                   void * hy, 
                                                   const hipdnnTensorDescriptor_t cyDesc, 
                                                   void * cy, 
                                                   void * workspace, 
                                                   size_t workSpaceSizeInBytes,
                                                   void * reserveSpace, 
                                                   size_t reserveSpaceSizeInBytes)
{
        return cudnnTohipdnnStatus(
	       cudnnRNNForwardTraining(handle,
                                        rnnDesc,
                                        seqLength,
                                        xDesc,
                                        x,
                                        hxDesc,
                                        hx,
                                        cxDesc,
                                        cx,
                                        wDesc,
                                        w,
                                        yDesc,
                                        y,
                                        hyDesc,
                                        hy,
                                        cyDesc,
                                        cy,
                                        workspace,
                                        workSpaceSizeInBytes,
                                        reserveSpace,
                                        reserveSpaceSizeInBytes));
}

hipdnnStatus_t  hipdnnRNNBackwardData( hipdnnHandle_t handle, 
                                                const hipdnnRNNDescriptor_t rnnDesc, 
                                                const int seqLength, 
                                                const hipdnnTensorDescriptor_t * yDesc, 
                                                const void * y,                                                
                                                const hipdnnTensorDescriptor_t * dyDesc, 
                                                const void * dy, 
                                                const hipdnnTensorDescriptor_t dhyDesc, 
                                                const void * dhy, 
                                                const hipdnnTensorDescriptor_t dcyDesc, 
                                                const void * dcy, 
                                                const hipdnnFilterDescriptor_t wDesc, 
                                                const void * w, 
                                                const hipdnnTensorDescriptor_t hxDesc, 
                                                const void * hx,                                                                  
                                                const hipdnnTensorDescriptor_t cxDesc, 
                                                const void * cx,                                                 
                                                const hipdnnTensorDescriptor_t * dxDesc, 
                                                void * dx, 
                                                const hipdnnTensorDescriptor_t dhxDesc,
                                                void * dhx,
                                                const hipdnnTensorDescriptor_t dcxDesc,
                                                void * dcx,
                                                void * workspace,
                                                size_t workSpaceSizeInBytes,
                                                void * reserveSpace, 
                                                size_t reserveSpaceSizeInBytes )
{
        return cudnnTohipdnnStatus(
		cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc,
					hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace,
					 reserveSpaceSizeInBytes));
}

hipdnnStatus_t  hipdnnRNNBackwardWeights( hipdnnHandle_t handle, 
                                                   const hipdnnRNNDescriptor_t rnnDesc, 
                                                   const int seqLength, 
                                                   const hipdnnTensorDescriptor_t * xDesc, 
                                                   const void * x, 
                                                   const hipdnnTensorDescriptor_t hxDesc, 
                                                   const void * hx,                                                   
                                                   const hipdnnTensorDescriptor_t * yDesc, 
                                                   const void * y,
                                                   const void * workspace, 
                                                   size_t workSpaceSizeInBytes, 
                                                   const hipdnnFilterDescriptor_t dwDesc, 
                                                   void * dw,
                                                   const void * reserveSpace, 
                                                   size_t reserveSpaceSizeInBytes )
{
        return cudnnTohipdnnStatus(
		cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workSpaceSizeInBytes, 
						dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnBatchNormalizationForwardInference(
                                hipdnnHandle_t                       handle,
                                hipdnnBatchNormMode_t                mode,
                                const void                         *alpha, // alpha[0] = result blend factor
                                const void                         *beta,  // beta[0] = dest layer blend factor
                                const hipdnnTensorDescriptor_t       xDesc,
                                const void                         *x,     // NxCxHxW
                                const hipdnnTensorDescriptor_t       yDesc,
                                void                               *y,     // NxCxHxW
                                const hipdnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                const void                         *bnScale,
                                const void                         *bnBias,
                                const void                         *estimatedMean,
                                const void                         *estimatedVariance,
                                double                              epsilon )
{

        return cudnnTohipdnnStatus(
                cudnnBatchNormalizationForwardInference(handle, hipTocudnnBatchNormMode(mode), alpha, beta, xDesc, x, yDesc,
                                                        y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon));
}


//CNTK 2.4 SUPPORT

hipdnnStatus_t hipdnnCreateReduceTensorDescriptor(
                                hipdnnReduceTensorDescriptor_t          *reduceTensorDesc )
{
	return cudnnTohipdnnStatus(
		cudnnCreateReduceTensorDescriptor(reduceTensorDesc));
}

hipdnnStatus_t hipdnnSetTensor4dDescriptorEx(
                                hipdnnTensorDescriptor_t             tensorDesc,
                                hipdnnDataType_t                     dataType, /* image data type */
                                int                                 n,        /* number of inputs (batch size) */
                                int                                 c,        /* number of input feature maps */
                                int                                 h,        /* height of input section */
                                int                                 w,        /* width of input section */
                                int                                 nStride,
                                int                                 cStride,
                                int                                 hStride,
                                int                                 wStride )
{
	cudnnDataType_t cuDT;
	hipdnnStatus_t retVal;
	retVal = hipTocudnnDataType(dataType, &cuDT);
	if(retVal == HIPDNN_STATUS_SUCCESS)
		return cudnnTohipdnnStatus(cudnnSetTensor4dDescriptorEx(tensorDesc, cuDT, n, c, h, w,
									nStride, cStride, hStride, wStride));
	return retVal;
}

hipdnnStatus_t hipdnnSetReduceTensorDescriptor(
                                hipdnnReduceTensorDescriptor_t           reduceTensorDesc,
                                hipdnnReduceTensorOp_t                   reduceTensorOp,
                                hipdnnDataType_t                     reduceTensorCompType,
                                hipdnnNanPropagation_t               reduceTensorNanOpt,
                                hipdnnReduceTensorIndices_t          reduceTensorIndices,
                                hipdnnIndicesType_t                  reduceTensorIndicesType )
{
    cudnnReduceTensorOp_t cuRTO;
    cudnnDataType_t cuDT;
    cudnnNanPropagation_t cuNP;
    cudnnReduceTensorIndices_t cuRTI;
    cudnnIndicesType_t cuIT;
    hipdnnStatus_t retVal;

    retVal = hipTocudnnReduceTensorOp(reduceTensorOp, &cuRTO);
    if(retVal == HIPDNN_STATUS_SUCCESS)
    {
        retVal = hipTocudnnDataType(reduceTensorCompType, &cuDT);
        if(retVal == HIPDNN_STATUS_SUCCESS)
        {
            retVal = hipTocudnnNanPropagation(reduceTensorNanOpt, &cuNP);
            if(retVal == HIPDNN_STATUS_SUCCESS)
            {
                retVal = hipTocudnnReduceTensorIndices(reduceTensorIndices, &cuRTI);
                if(retVal == HIPDNN_STATUS_SUCCESS)
                {
                    retVal = hipTocudnnIndicesType(reduceTensorIndicesType, &cuIT);
                    if(retVal == HIPDNN_STATUS_SUCCESS)
                    {
                        return cudnnTohipdnnStatus(cudnnSetReduceTensorDescriptor(reduceTensorDesc, cuRTO, cuDT, cuNP, cuRTI, cuIT));
                    }
                }
            }
        }
    }
    return retVal;
}

hipdnnStatus_t hipdnnGetReductionWorkspaceSize(
                                hipdnnHandle_t handle,
                                const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                const cudnnTensorDescriptor_t aDesc,
                                const cudnnTensorDescriptor_t cDesc,
                                size_t *sizeInBytes )
{
    return cudnnTohipdnnStatus(cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes));
}

hipdnnStatus_t hipdnnReduceTensor(
                        hipdnnHandle_t                       handle,
                        const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
                        void                               *indices,
                        size_t                              indicesSizeInBytes,
                        void                               *workspace,
                        size_t                              workspaceSizeInBytes,
                        const void                         *alpha,
                        const hipdnnTensorDescriptor_t       aDesc,
                        const void                         *A,
                        const void                         *beta,
                        const hipdnnTensorDescriptor_t       cDesc,
                        void                               *C )
{
    return cudnnTohipdnnStatus(cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
                                                    workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C));
}

hipdnnStatus_t hipdnnDestroyReduceTensorDescriptor(hipdnnReduceTensorDescriptor_t reduceTensorDesc )
{
    return cudnnTohipdnnStatus(cudnnDestroyReduceTensorDescriptor(reduceTensorDesc));
}



