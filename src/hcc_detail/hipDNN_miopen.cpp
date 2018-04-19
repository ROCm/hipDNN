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
#include <stdint.h>
#include <assert.h>
#include <map>

#define PROMOTE_TO_SUPPORTED

#define DEBUG_CALL_STACK_LEVEL_NONE 0
#define DEBUG_CALL_STACK_LEVEL_ERRORS 1
#define DEBUG_CALL_STACK_LEVEL_PROMOTED 2
#define DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC 2
#define DEBUG_CALL_STACK_LEVEL_CALLS 3
#define DEBUG_CALL_STACK_LEVEL_MARSHALLING 4
#define DEBUG_CURRENT_CALL_STACK_LEVEL DEBUG_CALL_STACK_LEVEL_MARSHALLING

#ifndef thread_local
#if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#define thread_local _Thread_local
#elif defined _WIN32 && ( \
      defined _MSC_VER || \
      defined __ICL || \
      defined __DMC__ || \
      defined __BORLANDC__ )
#define thread_local __declspec(thread) 
/* note that ICC (linux) and Clang are covered by __GNUC__ */
#elif defined __GNUC__ || \
       defined __SUNPRO_C || \
       defined __xlC__
#define thread_local __thread
#else
//#error "Cannot define thread_local"
// NOT THREAD SAFE.
#define thread_local
#endif
#endif

//HGSOS not implemented yet because i dont know how to get the device pointers from the desc!!!
static thread_local void* sConvolutionForwardAlgorithmWorkspace;
static thread_local void* sConvolutionBackwardDataAlgorithmWorkspace;
static thread_local void* sConvolutionBackwardFilterAlgorithmWorkspace;

//Eventually those can be merged, but currently i will not mix device pointers with host data in the same map...
//HGSOS static std::map<miopenPoolingDescriptor_t, void *>  sPoolingDescToWorkspace;  ????
static std::map<miopenTensorDescriptor_t, int8_t*> sDescToWorkspacePooling; //device pointers
static std::map<miopenTensorDescriptor_t, size_t> sDescToWorkspacePoolingSize; //host

static std::map<miopenTensorDescriptor_t, int8_t*> sDescToWorkspaceLRN; //device pointers
static std::map<miopenTensorDescriptor_t, size_t> sDescToWorkspaceLRNSize; //host

//=============================================================================

hipdnnStatus_t miopenTohipdnnStatus(miopenStatus_t cStatus)
{
    hipdnnStatus_t retVal;
    switch (cStatus)
    {
    case miopenStatusSuccess:
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case miopenStatusNotInitialized:
        retVal = HIPDNN_STATUS_NOT_INITIALIZED;
        break;
    case miopenStatusAllocFailed:
        retVal = HIPDNN_STATUS_ALLOC_FAILED;
        break;
    case miopenStatusBadParm:
        retVal = HIPDNN_STATUS_BAD_PARAM;
        break;
    case miopenStatusInternalError:
        retVal = HIPDNN_STATUS_INTERNAL_ERROR;
        break;
    case miopenStatusInvalidValue:
        retVal = HIPDNN_STATUS_INVALID_VALUE;
        break;
    case miopenStatusUnknownError:
        retVal = HIPDNN_STATUS_EXECUTION_FAILED;
        break;
    case miopenStatusNotImplemented:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    default:
        retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenDataType(hipdnnDataType_t in, miopenDataType_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    switch (in)
    {
    case HIPDNN_DATA_FLOAT:
        *out = miopenFloat;
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case HIPDNN_DATA_HALF:
        *out = miopenHalf;
        retVal = HIPDNN_STATUS_SUCCESS;
        break;
    case HIPDNN_DATA_DOUBLE:
    case HIPDNN_DATA_INT8:
    case HIPDNN_DATA_INT32:
    case HIPDNN_DATA_INT8x4:
        break;
    }

    return retVal;
}

hipdnnStatus_t miopenTohipDataType(miopenDataType_t in, hipdnnDataType_t* out)
{
    switch (in)
    {
    case miopenFloat:
        *out = HIPDNN_DATA_FLOAT;
        break;
    case miopenHalf:
        *out = HIPDNN_DATA_HALF;
        break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t miopenTohipOpTensorOp(miopenTensorOp_t in,
        hipdnnOpTensorOp_t* out)
{
    switch (in)
    {
    case miopenTensorOpAdd:
        *out = HIPDNN_OP_TENSOR_ADD;
        break;
    case miopenTensorOpMul:
        *out = HIPDNN_OP_TENSOR_MUL;
        break;
    case miopenTensorOpMin:
        *out = HIPDNN_OP_TENSOR_MIN;
        break;
    case miopenTensorOpMax:
        *out = HIPDNN_OP_TENSOR_MAX;
        break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipTomiopenOpTensorOp(hipdnnOpTensorOp_t in,
        miopenTensorOp_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in)
    {
    case HIPDNN_OP_TENSOR_ADD:
        *out = miopenTensorOpAdd;
        break;
    case HIPDNN_OP_TENSOR_MUL:
        *out = miopenTensorOpMul;
        break;
    case HIPDNN_OP_TENSOR_MIN:
        *out = miopenTensorOpMin;
        break;
    case HIPDNN_OP_TENSOR_MAX:
        *out = miopenTensorOpMax;
        break;
    case HIPDNN_OP_TENSOR_SQRT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

    return retVal;
}

//=============================================================================

hipdnnConvolutionMode_t miopenTohipConvolutionMode(miopenConvolutionMode_t in)
{
    if (in == miopenConvolution)
        return HIPDNN_CONVOLUTION;
    /*else if( in == miopenCrossCorrelation )
     return HIPDNN_CROSS_CORRELATION;*/ //TODO: to be added
    return HIPDNN_CONVOLUTION;
}

miopenConvolutionMode_t hipTomiopenConvolutionMode(hipdnnConvolutionMode_t in)
{
    if (in == HIPDNN_CONVOLUTION)
        return miopenConvolution;
    /*else if( in == HIPDNN_CROSS_CORRELATION )
     return miopenCrossCorrelation;*/ //TODO: to be added
    return miopenConvolution;
}

//=============================================================================

hipdnnStatus_t hipTomiopenPoolingMode(hipdnnPoolingMode_t in,
        miopenPoolingMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_POOLING_MAX:
        *out = miopenPoolingMax;
        break;
    case HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
        *out = miopenPoolingAverage;
        break;
    case HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
        *out = miopenPoolingAverage;
        break;
    case HIPDNN_POOLING_MAX_DETERMINISTIC:
        *out = miopenPoolingMax;
        break;
    }
    return retVal;
}

hipdnnStatus_t miopenTohipPoolingMode(miopenPoolingMode_t in,
        hipdnnPoolingMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case miopenPoolingMax:
        *out = HIPDNN_POOLING_MAX;
        break;
    case miopenPoolingAverage:
        *out = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
//HGSOS     *out = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
//HGSOS     *out = HIPDNN_POOLING_MAX_DETERMINISTIC;
    default:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "miopenTohipPoolingMode " << in << ": NOT SUPPORTED." << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenLRNMode(hipdnnLRNMode_t in, miopenLRNMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_LRN_WITHIN_CHANNEL:
        *out = miopenLRNWithinChannel;
        break;
    case HIPDNN_LRN_CROSS_CHANNEL:
        *out = miopenLRNCrossChannel;
        break;
    }
    return retVal;
}

hipdnnStatus_t miopenTohipLRNMode(miopenLRNMode_t in, hipdnnLRNMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case miopenLRNWithinChannel:
        *out = HIPDNN_LRN_WITHIN_CHANNEL;
        break;
    case miopenLRNCrossChannel:
        *out = HIPDNN_LRN_CROSS_CHANNEL;
        break;
    }
    return retVal;
}

//=============================================================================

miopenBatchNormMode_t hipTomiopenBatchNormMode(hipdnnBatchNormMode_t in)
{
    if (in == HIPDNN_BATCHNORM_PER_ACTIVATION)
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_BATCHNORM_PER_ACTIVATION" << std::endl;
#endif

        return miopenBNPerActivation;
    }
    else if (in == HIPDNN_BATCHNORM_SPATIAL)
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_BATCHNORM_SPATIAL" << std::endl;
#endif

        return miopenBNSpatial;
    }

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "HIPDNN_BATCHNORM_SPATIAL" << std::endl;
#endif

//HGSOS need to return error code, those are not the only options!
    return miopenBNPerActivation;
}

//=============================================================================

hipdnnStatus_t hipTomiopenActivationMode(hipdnnActivationMode_t in,
        miopenActivationMode_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in)
    {
    case HIPDNN_ACTIVATION_SIGMOID:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_SIGMOID" << std::endl;
#endif

        *out = miopenActivationLOGISTIC;
        break;

    case HIPDNN_ACTIVATION_RELU:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_RELU" << std::endl;
#endif

        *out = miopenActivationRELU;
        break;

    case HIPDNN_ACTIVATION_TANH:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        *out = miopenActivationTANH;
        break;

    case HIPDNN_ACTIVATION_PATHTRU:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        *out = miopenActivationPATHTRU;
        break;

    case HIPDNN_ACTIVATION_SOFTRELU:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        *out = miopenActivationSOFTRELU;
        break;

    case HIPDNN_ACTIVATION_ABS:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        *out = miopenActivationABS;
        break;

    case HIPDNN_ACTIVATION_POWER:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        *out = miopenActivationPOWER;
        break;

    case HIPDNN_ACTIVATION_ELU:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;

    case HIPDNN_ACTIVATION_CLIPPED_RELU:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;

    default:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "HIPDNN_ACTIVATION_TANH" << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionFwdAlgo(hipdnnConvolutionFwdAlgo_t in,
        miopenConvFwdAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in)
    {

    case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_FWD_ALGO_GEMM" << std::endl;
#endif

        *out = miopenConvolutionFwdAlgoGEMM;
        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT" << std::endl;
#endif
        *out = miopenConvolutionFwdAlgoDirect;

        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_FWD_ALGO_FFT" << std::endl;
#endif

        *out = miopenConvolutionFwdAlgoFFT;
        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD" << std::endl;
#endif
        *out = miopenConvolutionFwdAlgoWinograd;
        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM" << std::endl;
#endif

        *out = miopenConvolutionFwdAlgoGEMM;
        break;

    default:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout <<"hipdnnConvolutionFwdAlgo_t: " << in << " NOT SUPPORTED." << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

    return retVal;
}

hipdnnStatus_t miopenTohipConvolutionFwdAlgo(miopenConvFwdAlgorithm_t in,
        hipdnnConvolutionFwdAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {

    case miopenConvolutionFwdAlgoGEMM:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_GEMM;
        break;
    case miopenConvolutionFwdAlgoDirect:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        break;
    case miopenConvolutionFwdAlgoFFT:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;
        break;
    case miopenConvolutionFwdAlgoWinograd:
        *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        break;
    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

    return retVal;
}

int ConvolutionFwdAlgoCount()
{
    return 4;
}

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t GetConvolutionFwdAlgo(int i)
{
    hipdnnConvolutionFwdAlgo_t retVal;
    miopenConvFwdAlgorithm_t mialgo;

    if (i < ConvolutionFwdAlgoCount())
    {
        mialgo = (miopenConvFwdAlgorithm_t) i;
    }
    else
    {
        //for protection
        mialgo = (miopenConvFwdAlgorithm_t) miopenConvolutionFwdAlgoWinograd;
    }
    miopenTohipConvolutionFwdAlgo(mialgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionBwdFilterAlgo(
        hipdnnConvolutionBwdFilterAlgo_t in,
        miopenConvBwdWeightsAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0" << std::endl;
#endif

        *out = miopenConvolutionBwdWeightsAlgoGEMM;
        break;

    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1" << std::endl;
#endif

        *out = miopenConvolutionBwdWeightsAlgoDirect;

        break;
        /*case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:*/ //TODO: will be added in future
    default:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout <<"hipdnnConvolutionBwdFilterAlgo_t: " << in << " NOT SUPPORTED." << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

    return retVal;
}

hipdnnStatus_t miopenTohipConvolutionBwdFilterAlgo(
        miopenConvBwdWeightsAlgorithm_t in,
        hipdnnConvolutionBwdFilterAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case miopenConvolutionBwdWeightsAlgoGEMM:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        break;
    case miopenConvolutionBwdWeightsAlgoDirect:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        break;
    }

    return retVal;
}

int ConvolutionBwdFilterAlgoCount()
{
    return (int) 2;
}

// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t GetConvolutionBwdFilterAlgo(int i)
{
    hipdnnConvolutionBwdFilterAlgo_t retVal;
    miopenConvBwdWeightsAlgorithm_t mialgo;

    if (i < ConvolutionBwdFilterAlgoCount())
    {
        mialgo = (miopenConvBwdWeightsAlgorithm_t) i;
    }
    else
    {
        //for protection
        mialgo =
                (miopenConvBwdWeightsAlgorithm_t) miopenConvolutionBwdWeightsAlgoGEMM;
    }
    miopenTohipConvolutionBwdFilterAlgo(mialgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionBwdDataAlgo(
        hipdnnConvolutionBwdDataAlgo_t in, miopenConvBwdDataAlgorithm_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0" << std::endl;
#endif

        *out = miopenConvolutionBwdDataAlgoGEMM;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1" << std::endl;
#endif

        *out = miopenConvolutionBwdDataAlgoDirect;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD" << std::endl;
#endif

        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
                << std::endl;
#endif

        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT" << std::endl;
#endif

        *out = miopenConvolutionBwdDataAlgoFFT;
        break;

        /*case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
         case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:*/ //TODO: to be added in future

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM" << std::endl;
#endif

        *out = miopenTransposeBwdDataAlgoGEMM;
        break;

    default:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout <<"hipdnnConvolutionBwdDataAlgo_t: " << in << " NOT SUPPORTED." << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
}

hipdnnStatus_t miopenTohipConvolutionBwdDataAlgo(
        miopenConvBwdDataAlgorithm_t in, hipdnnConvolutionBwdDataAlgo_t* out)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case miopenConvolutionBwdDataAlgoGEMM:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        break;
    case miopenConvolutionBwdDataAlgoDirect:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        break;
    case miopenConvolutionBwdDataAlgoFFT:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
        break;
    case miopenConvolutionBwdDataAlgoWinograd:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
    case miopenTransposeBwdDataAlgoGEMM:
        *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM;
        break;
    }
    return retVal;
}

int ConvolutionBwdDataAlgoCount()
{
    return (int) 2;
}

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t GetConvolutionBwdDataAlgo(int i)
{
    hipdnnConvolutionBwdDataAlgo_t retVal;
    miopenConvBwdDataAlgorithm_t mialgo;

    if (i < ConvolutionBwdDataAlgoCount())
    {
        mialgo = (miopenConvBwdDataAlgorithm_t) i;
    }
    else
    {
        //for protection
        mialgo =
                (miopenConvBwdDataAlgorithm_t) miopenConvolutionBwdDataAlgoWinograd;
    }
    miopenTohipConvolutionBwdDataAlgo(mialgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipSoftmaxModeSupported(hipdnnSoftmaxMode_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    //HGSOS
    case HIPDNN_SOFTMAX_MODE_INSTANCE:

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "HIPDNN_SOFTMAX_MODE_INSTANCE NOT SUPPORTED." << std::endl;
#endif

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_SOFTMAX_MODE_CHANNEL:
        break;
    }
    return retVal;
}

hipdnnStatus_t SoftmaxAlgorithmSupported(hipdnnSoftmaxAlgorithm_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_SOFTMAX_FAST:
    case HIPDNN_SOFTMAX_ACCURATE:
        break;
    case HIPDNN_SOFTMAX_LOG:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
}

// miopen does not define tensor format, 
// implicitly HIPDNN_TENSOR_NCHW only
hipdnnStatus_t hipTensorFormatSupported(hipdnnTensorFormat_t in)
{
    if (in == HIPDNN_TENSOR_NCHW)
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_MARSHALLING
        std::cout << "HIPDNN_TENSOR_NCHW" << std::endl;
#endif

        return HIPDNN_STATUS_SUCCESS;
    }

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout <<"hipdnnTensorFormat_t " << in << " NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t ConvolutionFwdPreferenceSupported(
        hipdnnConvolutionFwdPreference_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST:
        break;
    case HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    if(retVal == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        std::cout <<"hipdnnConvolutionFwdPreference_t " << in << " NOT SUPPORTED." << std::endl;
    }
#endif

    return retVal;
}

hipdnnStatus_t ConvolutionBwdFilterPreferenceSupported(
        hipdnnConvolutionBwdFilterPreference_t in)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in)
    {
    case HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    if(retVal == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        std::cout <<"hipdnnConvolutionBwdFilterPreference_t " << in << " NOT SUPPORTED." << std::endl;
    }
#endif

    return retVal;
}

//=============================================================================

HIPDNN_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle)
{
    sConvolutionForwardAlgorithmWorkspace = 0;
    sConvolutionBackwardDataAlgorithmWorkspace = 0;
    sConvolutionBackwardFilterAlgorithmWorkspace = 0;
    return miopenTohipdnnStatus(miopenCreate(handle));
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle)
{
    if (sConvolutionForwardAlgorithmWorkspace != 0)
    {
        hipFree(sConvolutionForwardAlgorithmWorkspace);
        sConvolutionForwardAlgorithmWorkspace = 0;
    }
    if (sConvolutionBackwardDataAlgorithmWorkspace != 0)
    {
        hipFree(sConvolutionBackwardDataAlgorithmWorkspace);
        sConvolutionBackwardDataAlgorithmWorkspace = 0;
    }
    if (sConvolutionBackwardFilterAlgorithmWorkspace != 0)
    {
        hipFree(sConvolutionBackwardFilterAlgorithmWorkspace);
        sConvolutionBackwardFilterAlgorithmWorkspace = 0;
    }

    return miopenTohipdnnStatus(miopenDestroy(handle));
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId)
{
    return miopenTohipdnnStatus(miopenSetStream(handle, streamId));
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipdnnStream_t *streamId)
{
    return miopenTohipdnnStatus(miopenGetStream(handle, streamId));
}

size_t hipdnnGetVersion()
{
    return 6000;
}

hipdnnStatus_t hipdnnCreateTensorDescriptor(
        hipdnnTensorDescriptor_t *tensorDesc)
{
    return miopenTohipdnnStatus(miopenCreateTensorDescriptor(tensorDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnSetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
        hipdnnTensorFormat_t format, hipdnnDataType_t dataType, int n, int c,
        int h, int w)
{

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t miDT;

    retVal = hipTensorFormatSupported(format);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = hipTomiopenDataType(dataType, &miDT);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return miopenTohipdnnStatus(
            miopenSet4dTensorDescriptor(tensorDesc, miDT, n, c, h, w));

}

//=============================================================================

hipdnnStatus_t hipdnnGetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
        hipdnnDataType_t *dataType, int *n, int *c, int *h, int *w,
        int *nStride, int *cStride, int *hStride, int *wStride)
{
    miopenDataType_t midT;
    hipdnnStatus_t retVal;

    retVal = miopenTohipdnnStatus(
            miopenGet4dTensorDescriptor(tensorDesc, &midT, n, c, h, w, nStride,
                    cStride, hStride, wStride));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return miopenTohipDataType(midT, dataType);
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyTensorDescriptor(
        hipdnnTensorDescriptor_t tensorDesc)
{
    return miopenTohipdnnStatus(miopenDestroyTensorDescriptor(tensorDesc));
}

//=============================================================================

//dstValue = alpha[0]*srcValue + beta[0]*priorDstValue

hipdnnStatus_t hipdnnAddTensor(hipdnnHandle_t handle, const void *alpha,
        const hipdnnTensorDescriptor_t aDesc, const void *A, const void *beta,
        const hipdnnTensorDescriptor_t cDesc, void *C)
{
    miopenTensorOp_t tensorOp = miopenTensorOpAdd;
    int alpha2 = 0;

    return miopenTohipdnnStatus(
            miopenOpTensor(handle, tensorOp, alpha, aDesc, A, beta, cDesc, C,
                    &alpha2, cDesc, C));
}

//=============================================================================

hipdnnStatus_t hipdnnOpTensor(hipdnnHandle_t handle,
        const hipdnnOpTensorDescriptor_t opTensorDesc, const void *alpha1,
        const hipdnnTensorDescriptor_t aDesc, const void *A, const void *alpha2,
        const hipdnnTensorDescriptor_t bDesc, const void *B, const void *beta,
        const hipdnnTensorDescriptor_t cDesc, void *C)
{

    return miopenTohipdnnStatus(
            miopenOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2,
                    bDesc, B, beta, cDesc, C));
}
//=============================================================================

hipdnnStatus_t hipdnnSetTensor(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t yDesc, void *y, const void *valuePtr)
{
    return miopenTohipdnnStatus(miopenSetTensor(handle, yDesc, y, valuePtr));
}

//=============================================================================

hipdnnStatus_t hipdnnScaleTensor(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t yDesc, void *y, const void *alpha)
{
    return miopenTohipdnnStatus(miopenScaleTensor(handle, yDesc, y, alpha));
}

//=============================================================================

hipdnnStatus_t hipdnnCreateFilterDescriptor(
        hipdnnFilterDescriptor_t *filterDesc)
{
    //in miopen a filter descriptor is just a typedef to a tensor descriptor
    return hipdnnCreateTensorDescriptor(filterDesc);
}

//=============================================================================

hipdnnStatus_t hipdnnCreateConvolutionDescriptor(
        hipdnnConvolutionDescriptor_t *convDesc)
{
    return miopenTohipdnnStatus(miopenCreateConvolutionDescriptor(convDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnSetConvolutionMathType(
        hipdnnConvolutionDescriptor_t convDesc, hipdnnMathType_t mathType)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout <<"hipdnnSetConvolutionMathType " << mathType << " NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

//=============================================================================

hipdnnStatus_t hipdnnSetConvolution2dDescriptor(
        hipdnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u,
        int v, int upscalex, int upscaley, hipdnnConvolutionMode_t mode,
        hipdnnDataType_t computeType)
{

    return miopenTohipdnnStatus(
            miopenInitConvolutionDescriptor(convDesc,
                    hipTomiopenConvolutionMode(mode), pad_h, pad_w, u, v,
                    upscalex, upscaley));
}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolution2dDescriptor(
        const hipdnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_y,
        int *u, int *v, int *upscalex, int *upscaley,
        hipdnnConvolutionMode_t *mode, hipdnnDataType_t *computeType)
{

    hipdnnStatus_t retVal;
    miopenConvolutionMode_t miMode;

    retVal = miopenTohipdnnStatus(
            miopenGetConvolutionDescriptor(convDesc, &miMode, pad_h, pad_y, u,
                    v, upscalex, upscaley));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    *mode = miopenTohipConvolutionMode(miMode);

    //HGSOS miopen does not support this. Any better way to do this?
    //which call should define the type?
    *computeType = HIPDNN_DATA_FLOAT;

    return retVal;
}

//===========

hipdnnStatus_t hipdnnGetConvolution2dForwardOutputDim(
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t inputTensorDesc,
        const hipdnnFilterDescriptor_t filterDesc, int *n, int *c, int *h,
        int *w)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "HIPDNN_SOFTMAX_MODE_INSTANCE NOT SUPPORTED." << std::endl;
#endif

    return miopenTohipdnnStatus(miopenGetConvolutionForwardOutputDim(convDesc, //HGSOSOS should be const in miopen.
            inputTensorDesc, filterDesc, n, c, h, w));
}

//==============================================================================

hipdnnStatus_t hipdnnDestroyConvolutionDescriptor(
        hipdnnConvolutionDescriptor_t convDesc)
{
    return miopenTohipdnnStatus(miopenDestroyConvolutionDescriptor(convDesc));
}

//===============================================================================

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithm(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
        int *returnedAlgoCount, hipdnnConvolutionFwdAlgoPerf_t *perfResults)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "hipdnnFindConvolutionForwardAlgorithm NOT IMPLEMENTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
#ifdef NOTYET   

    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal;

    //in miopen, workspace size does not depend on algo.
    retVal = miopenTohipdnnStatus(
            miopenConvolutionForwardGetWorkSpaceSize( handle,
                    wDesc,
                    xDesc,
                    convDesc,
                    yDesc,
                    sizeInBytes));

    if( retVal != HIPDNN_STATUS_SUCCESS)
    return retVal;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC hipdnnFindConvolutionForwardAlgorithm\n";
#endif

    hipMalloc((void**)&sConvolutionForwardAlgorithmWorkspace, sizeInBytes);

    //HGSOS //NOTYET dont know how to get x,y,w from the descriptors but it should be possible.

    return hipdnnFindConvolutionForwardAlgorithmEx( handle,
            xDesc,
            const void *x,
            wDesc,
            const void *w,
            convDesc,
            yDesc,
            void *y,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
            void *workSpace,
            size_t workSpaceSizeInBytes)
#endif                                              
}

hipdnnStatus_t hipdnnGetConvolutionForwardAlgorithm(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t yDesc,
        hipdnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
        hipdnnConvolutionFwdAlgo_t *algo)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "hipdnnGetConvolutionForwardAlgorithm NOT IMPLEMENTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
#ifdef NOTYET   

    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal;

    //in miopen, workspace size does not depend on algo.
    retVal = miopenTohipdnnStatus(
            miopenConvolutionForwardGetWorkSpaceSize( handle,
                    wDesc,
                    xDesc,
                    convDesc,
                    yDesc,
                    sizeInBytes));

    if( retVal != HIPDNN_STATUS_SUCCESS)
    return retVal;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC hipdnnGetConvolutionForwardAlgorithm\n";
#endif

    hipMalloc((void**)&sConvolutionForwardAlgorithmWorkspace, sizeInBytes);

//HGSOS //NOTYET dont know how to get x,y,w from the descriptors but it should be possible.

    return hipdnnFindConvolutionForwardAlgorithmEx( handle,
            xDesc,
            const void *x,
            wDesc,
            const void *w,
            convDesc,
            yDesc,
            void *y,
            requestedAlgoCount,
            returnedAlgoCount,
            perfResults,
            void *workSpace,
            size_t workSpaceSizeInBytes)
#endif  
}

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithmEx(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnFilterDescriptor_t wDesc, const void *w,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t yDesc, void *y,
        const int requestedAlgoCount, int *returnedAlgoCount,
        hipdnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
        size_t workSpaceSizeInBytes)
{
    assert(x);
    assert(w);
    assert(y);
    if (workSpace == NULL || workSpaceSizeInBytes == 0)
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Inside hipdnnFindConvolutionForwardAlgorithmEx: "
                << (workSpace == NULL) << ", " << workSpaceSizeInBytes
                << std::endl;
#endif

        size_t size;
        hipdnnStatus_t retVal;
        retVal = miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                        convDesc, yDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {
            return retVal;
        }

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnFindConvolutionForwardAlgorithmEx size " << size
                << " requested AlgoCount: " << requestedAlgoCount << std::endl;
#endif

        hipMalloc((void**) &sConvolutionForwardAlgorithmWorkspace, size);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
    std::cout << "INTERNAL_ALLOC: sConvolutionForwardAlgorithmWorkspace "
            << "WSP= " << sConvolutionForwardAlgorithmWorkspace
            << " size = " << size << std::endl;
#endif

        miopenConvAlgoPerf_t miopenPerfResults[1];

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Size of miopenPerfResults " << sizeof(miopenPerfResults) << std::endl;
#endif

        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc,
                        w, convDesc, yDesc, y, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults,
                        sConvolutionForwardAlgorithmWorkspace, size, true //exhaustiveSearch
                        ));

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {
            return retVal;
        }

        miopenTohipConvolutionFwdAlgo(miopenPerfResults->fwd_algo,
                &(perfResults->algo));

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "hipdnnFindConvolutionForwardAlgorithmEx exits\n";
#endif

        return retVal;

    }
    else
    {
        hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

        miopenConvAlgoPerf_t miopenPerfResults[1];

        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc,
                        w, convDesc, yDesc, y, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults, workSpace,
                        workSpaceSizeInBytes, true //exhaustiveSearch
                        ));

        miopenTohipConvolutionFwdAlgo(miopenPerfResults->fwd_algo,
                &(perfResults->algo));

        return retVal;
    }
}

//=========================================!

hipdnnStatus_t hipdnnGetConvolutionForwardWorkspaceSize(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t yDesc, hipdnnConvolutionFwdAlgo_t algo,
        size_t *sizeInBytes)
{
    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    //in miopen, workspace size does not depend on algo.
    return miopenTohipdnnStatus(
            miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                    convDesc, yDesc, sizeInBytes));

}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionForward(hipdnnHandle_t handle,
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnFilterDescriptor_t wDesc, const void *w,
        const hipdnnConvolutionDescriptor_t convDesc,
        hipdnnConvolutionFwdAlgo_t algo, void *workSpace,
        size_t workSpaceSizeInBytes, const void *beta,
        const hipdnnTensorDescriptor_t yDesc, void *y)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "calling hipdnnConvolutionForward." << std::endl;
#endif

    if (workSpace == NULL || workSpaceSizeInBytes == 0)
    {
        // Allocate sConvolutionForwardAlgorithmWorkspace to gather work space value
        size_t size;
        hipdnnStatus_t retVal;


#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnConvolutionForward size." << std::endl;
#endif

        retVal = miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                        convDesc, yDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

        hipMalloc((void**) &sConvolutionForwardAlgorithmWorkspace, size);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
    std::cout << "INTERNAL_ALLOC: sConvolutionForwardAlgorithmWorkspace "
            << "WSP= " << sConvolutionForwardAlgorithmWorkspace
            << " size = " << size << std::endl;
#endif

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Invoking hipToMopenConvolutionFwdAlgo" << std::endl;
        std::cout << "Passed algo" << algo << std::endl;
#endif
        miopenConvFwdAlgorithm_t mialgo;
        retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Invoked hipToMopenConvolutionFwdAlgo" << std::endl;
#endif

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Invoking MiopenConvolutionFwd" << std::endl;
#endif

        return miopenTohipdnnStatus(
                miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w,
                        convDesc, mialgo, beta, yDesc, y,
                        sConvolutionForwardAlgorithmWorkspace, size));

    }
    else
    {

        miopenConvFwdAlgorithm_t mialgo;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Invoking hipToMopenConvolutionFwdAlgo" << std::endl;
        std::cout << "Passed algo" << algo << std::endl;
#endif

        hipdnnStatus_t retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Invoked hipToMopenConvolutionFwdAlgo" << std::endl;
#endif

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Invoking MiopenConvolutionFwd" << std::endl;
#endif

        return miopenTohipdnnStatus(
                miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w,
                        convDesc, mialgo, beta, yDesc, y, workSpace,
                        workSpaceSizeInBytes));
    }

}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardBias(hipdnnHandle_t handle,
        const void *alpha, const hipdnnTensorDescriptor_t dyDesc,
        const void *dy, const void *beta, const hipdnnTensorDescriptor_t dbDesc,
        void *db)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "calling hipdnnConvolutionBackwardBias." << std::endl;
#endif

    return miopenTohipdnnStatus(
            miopenConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta,
                    dbDesc, db));
}

//================HGSOS======================
hipdnnStatus_t hipdnnFindConvolutionBackwardFilterAlgorithm(
        hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
        int *returnedAlgoCount,
        hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "hipdnnFindConvolutionBackwardFilterAlgorithm NOT IMPLEMENTED" << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif

}

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterAlgorithm(
        hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnFilterDescriptor_t dwDesc,
        hipdnnConvolutionBwdFilterPreference_t preference,
        size_t memoryLimitInBytes, hipdnnConvolutionBwdFilterAlgo_t *algo)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "hipdnnGetConvolutionBackwardFilterAlgorithm NOT IMPLEMENTED" << std::endl;
#endif

#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnFindConvolutionBackwardFilterAlgorithmEx(
        hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
        const void *x, const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnFilterDescriptor_t dwDesc, void *dw,
        const int requestedAlgoCount, int *returnedAlgoCount,
        hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
        size_t workSpaceSizeInBytes)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "Inside hipdnnFindConvolutionBackwardFilterAlgorithmEx\n";
#endif

    assert(x);
    assert(dy);
    assert(dw);

    if (workSpace == NULL || workSpaceSizeInBytes == 0)
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC hipdnnFindConvolutionBackwardFilterAlgorithmEx\n";
#endif

        size_t size;
        hipdnnStatus_t retVal;
        retVal = miopenTohipdnnStatus(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc,
                        xDesc, convDesc, dwDesc, &size));
        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "miopenConvolutionBackwardGetWorkSpaceSize size " << size
                << " requested AlgoCount: " << requestedAlgoCount << std::endl;
#endif

        hipMalloc((void**) &sConvolutionBackwardFilterAlgorithmWorkspace, size);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
    std::cout << "INTERNAL_ALLOC: sConvolutionBackwardFilterAlgorithmWorkspace "
            << "WSP= " << sConvolutionBackwardFilterAlgorithmWorkspace
            << " size = " << size << std::endl;
#endif

        miopenConvAlgoPerf_t miopenPerfResults[1];

        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionBackwardWeightsAlgorithm(handle, dyDesc,
                        dy, xDesc, x, convDesc, dwDesc, dw, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults,
                        sConvolutionBackwardFilterAlgorithmWorkspace, size, true //exhaustiveSearch
                        ));
        miopenTohipConvolutionBwdFilterAlgo(miopenPerfResults->bwd_weights_algo,
                &(perfResults->algo));

        return retVal;

    }
    else
    {
        hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

        miopenConvAlgoPerf_t* miopenPerfResults = new miopenConvAlgoPerf_t;

        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionBackwardWeightsAlgorithm(handle, dyDesc,
                        dy, xDesc, x, convDesc, dwDesc, dw, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults, workSpace,
                        workSpaceSizeInBytes, true //exhaustiveSearch
                        ));

        miopenTohipConvolutionBwdFilterAlgo(miopenPerfResults->bwd_weights_algo,
                &(perfResults->algo));

        return retVal;

    }

}

//=================HGSOS======================!

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterWorkspaceSize(
        hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnFilterDescriptor_t dwDesc,
        hipdnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes)
{

// in miopen, workspace size doesnt depend on algo.

    return miopenTohipdnnStatus(
            miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc,
                    xDesc, convDesc, dwDesc, sizeInBytes));

}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardFilter(hipdnnHandle_t handle,
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnConvolutionDescriptor_t convDesc,
        hipdnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
        size_t workSpaceSizeInBytes, const void *beta,
        const hipdnnFilterDescriptor_t dwDesc, void *dw)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "CALL_STACK: Inside hipdnnConvolutionBackwardFilter\n";
#endif

    if (workSpaceSizeInBytes == 0 || workSpace == NULL)
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnConvolutionBackwardFilter\n";
#endif

        size_t size;
        hipdnnStatus_t retVal;
        retVal = miopenTohipdnnStatus(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc,
                        xDesc, convDesc, dwDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

        hipMalloc((void**) &sConvolutionBackwardFilterAlgorithmWorkspace, size);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
    std::cout << "INTERNAL_ALLOC: sConvolutionBackwardFilterAlgorithmWorkspace "
            << "WSP= " << sConvolutionBackwardFilterAlgorithmWorkspace
            << " size = " << size << std::endl;
#endif

        miopenConvBwdWeightsAlgorithm_t mialgo;
        retVal = hipTomiopenConvolutionBwdFilterAlgo(algo, &mialgo);

        return miopenTohipdnnStatus(
                miopenConvolutionBackwardWeights(handle, alpha, dyDesc, dy,
                        xDesc, x, convDesc, mialgo, beta, dwDesc, dw,
                        sConvolutionBackwardFilterAlgorithmWorkspace, size));

    }
    else
    {

        miopenConvBwdWeightsAlgorithm_t mialgo;
        hipdnnStatus_t retVal = hipTomiopenConvolutionBwdFilterAlgo(algo,
                &mialgo);

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

        return miopenTohipdnnStatus(
                miopenConvolutionBackwardWeights(handle, alpha, dyDesc, dy,
                        xDesc, x, convDesc, mialgo, beta, dwDesc, dw, workSpace,
                        workSpaceSizeInBytes));
    }

}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolutionBackwardDataWorkspaceSize(
        hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc,
        hipdnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes)
{
    //does not depend on algo in miopen

    return miopenTohipdnnStatus(
            miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc,
                    convDesc, dxDesc, sizeInBytes));
}

//=============================================================================

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithm(hipdnnHandle_t handle,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
        int *returnedAlgoCount, hipdnnConvolutionBwdDataAlgoPerf_t *perfResults)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "ERROR: hipdnnFindConvolutionBackwardDataAlgorithm NOT IMPLEMENTED" << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif

}

hipdnnStatus_t hipdnnGetConvolutionBackwardDataAlgorithm(hipdnnHandle_t handle,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc,
        hipdnnConvolutionBwdDataPreference_t preference,
        size_t memoryLimitInBytes, hipdnnConvolutionBwdDataAlgo_t *algo)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "ERROR: hipdnnGetConvolutionBackwardDataAlgorithm NOT IMPLEMENTED" << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif
}

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithmEx(
        hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
        const void *w, const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc, void *dx,
        const int requestedAlgoCount, int *returnedAlgoCount,
        hipdnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
        size_t workSpaceSizeInBytes)

{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnFindConvolutionBackwardDataAlgorithmEx: input ws size="
            << workSpaceSizeInBytes << ", requestedAlgoCount="
            << requestedAlgoCount << ", WS PTR=" << workSpace << std::endl;
#endif

    size_t size;
    hipdnnStatus_t retVal;
    retVal = miopenTohipdnnStatus(
            miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc,
                    convDesc, dxDesc, &size));

    if (retVal != HIPDNN_STATUS_SUCCESS)
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "...failed miopenConvolutionBackwardDataGetWorkSpaceSize" << std::endl;
#endif

        return retVal;
    }

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    if( size == 0 )
    {
        std::cout << "...zero WS size" << std::endl;
        return HIPDNN_STATUS_NOT_INITIALIZED;
    }
#endif

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
    std::cout << "INTERNAL_ALLOC: miopenConvolutionBackwardGetWorkSpaceSize size " << size
            << " requested AlgoCount: " << requestedAlgoCount << std::endl;
#endif

    hipMalloc((void**) &sConvolutionBackwardDataAlgorithmWorkspace, size);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
    std::cout << "INTERNAL_ALLOC: miopenConvolutionBackwardGetWorkSpaceSize "
            << "WSP= " << sConvolutionBackwardDataAlgorithmWorkspace
            << " size = " << size << std::endl;
#endif

    miopenConvAlgoPerf_t* miopenPerfResults = new miopenConvAlgoPerf_t[requestedAlgoCount];

    retVal = miopenTohipdnnStatus(
            miopenFindConvolutionBackwardDataAlgorithm(handle, dyDesc, dy,
                    wDesc, w, convDesc, dxDesc, dx, requestedAlgoCount,
                    returnedAlgoCount,
                    miopenPerfResults,
                    sConvolutionBackwardDataAlgorithmWorkspace, size, true // exhaustiveSearch
                    ));


    for (int i = 0; i < *returnedAlgoCount; i++)
    {
        std::cout << "miopenPerfResults:"
                << " algo = " << miopenPerfResults[i].bwd_data_algo
                << " time = " << miopenPerfResults[i].time
                << " memory = " << miopenPerfResults[i].memory << std::endl ;
    }

    if( retVal != HIPDNN_STATUS_SUCCESS)
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "...failed miopenFindConvolutionBackwardDataAlgorithm, returnedAlgoCount:"
                << *returnedAlgoCount << std::endl;
#endif
    }
    else
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "...miopenFindConvolutionBackwardDataAlgorithm OK, returnedAlgoCount:"
                << *returnedAlgoCount << std::endl;
#endif
    }

    for (int i = 0; i < *returnedAlgoCount; i++)
    {
        retVal = miopenTohipConvolutionBwdDataAlgo(
                    miopenPerfResults[i].bwd_data_algo,
                    &(perfResults[i].algo));

        if( retVal != HIPDNN_STATUS_SUCCESS)
        {
    #if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
            std::cout << "...failed miopenTohipConvolutionBwdDataAlgo" << std::endl;
    #endif
        }
        else
        {
    #if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
            std::cout << "...miopenTohipConvolutionBwdDataAlgo OK" << std::endl;
    #endif
        }
    }

    delete [] miopenPerfResults;
//HGSOS    workSpace = sConvolutionBackwardDataAlgorithmWorkspace;

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardData(hipdnnHandle_t handle,
        const void *alpha, const hipdnnFilterDescriptor_t wDesc, const void *w,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnConvolutionDescriptor_t convDesc,
        hipdnnConvolutionBwdDataAlgo_t algo, void *workSpace,
        size_t workSpaceSizeInBytes, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "ConvolutionBackwardData: WS PTR="
            << workSpace << ", WS size = " << workSpaceSizeInBytes << std::endl;
#endif

    if (workSpace == NULL || workSpaceSizeInBytes == 0)
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "ConvolutionBackwardData: INTERNAL_ALLOC: hipdnnConvolutionBackwardData\n";
#endif

        size_t size;
        hipdnnStatus_t retVal;
        retVal = miopenTohipdnnStatus(
                miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc,
                        wDesc, convDesc, dxDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "ConvolutionBackwardData: failed miopenConvolutionBackwardDataGetWorkSpaceSize: " << retVal << std::endl;
#endif

            return retVal;
        }

        hipMalloc((void**) &sConvolutionBackwardDataAlgorithmWorkspace, size);

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "ConvolutionBackwardData: Allocated workspace "
                << "WSP = " << sConvolutionBackwardDataAlgorithmWorkspace << ", size:" << size
                << std::endl;
#endif

        // Allocate sConvolutionBackwardDataAlgorithmWorkspace to gather work space value
        miopenConvBwdDataAlgorithm_t mialgo;
        retVal = hipTomiopenConvolutionBwdDataAlgo(algo, &mialgo);

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "ConvolutionBackwardData: failed hipTomiopenConvolutionBwdDataAlgo: "
            "HIP algo = " << algo << ", miopen algo= " << mialgo << ", error = "<< retVal << std::endl;
#endif

            return retVal;
        }
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "ConvolutionBackwardData:  hipTomiopenConvolutionBwdDataAlgo OK." << std::endl;
        std::cout << "ConvolutionBackwardData: about to invoke miopenConvolutionBackwardData."
                << ", WS PTR = " << sConvolutionBackwardDataAlgorithmWorkspace << ", WS size =" << size << std::endl;
#endif

        return miopenTohipdnnStatus(
                miopenConvolutionBackwardData(handle, alpha, dyDesc, dy, wDesc,
                        w, convDesc, mialgo, beta, dxDesc, dx,
                        sConvolutionBackwardDataAlgorithmWorkspace, size));

    }
    else
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "ConvolutionBackwardData: workspace preallocated." << std::endl;
#endif

        miopenConvBwdDataAlgorithm_t mialgo;
        hipdnnStatus_t retVal = hipTomiopenConvolutionBwdDataAlgo(algo, &mialgo);

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "ConvolutionBackwardData: failed hipTomiopenConvolutionBwdDataAlgo: "
            "HIP algo = " << algo << ", miopen algo= " << mialgo << ", error = "<< retVal << std::endl;
#endif
            return retVal;
        }

        float a = *(static_cast<const float*>(alpha));
        float b = *(static_cast<const float*>(beta));

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "ConvolutionBackwardData: alpha and beta values are " << a << " and " << b
                << std::endl;
#endif

        return miopenTohipdnnStatus(
                miopenConvolutionBackwardData(handle, alpha, dyDesc, dy, wDesc,
                        w, convDesc, mialgo, beta, dxDesc, dx, workSpace,
                        workSpaceSizeInBytes));

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
        std::cout << "ConvolutionBackwardData: Invoked miopenConvolutionBackwardData " << std::endl;
#endif


    }
}

//=============================================================================

hipdnnStatus_t hipdnnSoftmaxForward(hipdnnHandle_t handle,
        hipdnnSoftmaxAlgorithm_t algo, hipdnnSoftmaxMode_t mode,
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y)
{

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSoftmaxForward\n";
#endif

    retVal = SoftmaxAlgorithmSupported(algo);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = hipSoftmaxModeSupported(mode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return miopenTohipdnnStatus(
            miopenSoftmaxForward(handle, alpha, xDesc, x, beta, yDesc, y));
}

//=============================================================================

hipdnnStatus_t hipdnnSoftmaxBackward(hipdnnHandle_t handle,
        hipdnnSoftmaxAlgorithm_t algo, hipdnnSoftmaxMode_t mode,
        const void *alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSoftmaxBackward\n";
#endif

    retVal = SoftmaxAlgorithmSupported(algo);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = hipSoftmaxModeSupported(mode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return miopenTohipdnnStatus(
            miopenSoftmaxBackward(handle, alpha, yDesc, y, dyDesc, dy, beta,
                    dxDesc, dx));
}

//=============================================================================

hipdnnStatus_t hipdnnCreatePoolingDescriptor(
        hipdnnPoolingDescriptor_t *poolingDesc)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreatePoolingDescriptor\n";
#endif

    retVal = miopenTohipdnnStatus(miopenCreatePoolingDescriptor(poolingDesc));

    return retVal;
}
//=============================================================================

hipdnnStatus_t hipdnnSetPooling2dDescriptor(
        hipdnnPoolingDescriptor_t poolingDesc, hipdnnPoolingMode_t mode,
        hipdnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
        int windowWidth, int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride)
{

    miopenPoolingMode_t miPMode;
    hipdnnStatus_t retVal;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetPooling2dDescriptor\n";
#endif

    retVal = hipTomiopenPoolingMode(mode, &miPMode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    //HGSOS verify

    return miopenTohipdnnStatus(
            miopenSet2dPoolingDescriptor(poolingDesc, miPMode, windowHeight,
                    windowWidth, horizontalPadding, verticalPadding,
                    horizontalStride, verticalStride));

}

//=============================================================================

hipdnnStatus_t hipdnnGetPooling2dDescriptor(
        const hipdnnPoolingDescriptor_t poolingDesc, hipdnnPoolingMode_t *mode,
        hipdnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
        int *windowWidth, int *verticalPadding, int *horizontalPadding,
        int *verticalStride, int *horizontalStride)
{
    hipdnnStatus_t retVal;
    miopenPoolingMode_t mipmmode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnGetPooling2dDescriptor\n";
#endif

    retVal = miopenTohipdnnStatus(
            miopenGet2dPoolingDescriptor(poolingDesc, &mipmmode, windowHeight,
                    windowWidth, horizontalPadding, horizontalPadding,
                    horizontalStride, verticalStride));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    *maxpoolingNanOpt = HIPDNN_PROPAGATE_NAN;

    return miopenTohipPoolingMode(mipmmode, mode);
}

//=============================================================================

hipdnnStatus_t hipdnnGetPooling2dForwardOutputDim(
        const hipdnnPoolingDescriptor_t poolingDesc,
        const hipdnnTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h,
        int *w)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnGetPooling2dDescriptor\n";
#endif

    return miopenTohipdnnStatus(
            miopenGetPoolingForwardOutputDim(poolingDesc, inputTensorDesc, n, c,
                    h, w));
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyPoolingDescriptor(
        hipdnnPoolingDescriptor_t poolingDesc)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDestroyPoolingDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenDestroyPoolingDescriptor(poolingDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingForward(hipdnnHandle_t handle,
        const hipdnnPoolingDescriptor_t poolingDesc, const void *alpha,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t yDesc, void *y)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;
    int8_t* devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnPoolingForward\n";
#endif

    if (sDescToWorkspacePooling.find(yDesc) != sDescToWorkspacePooling.end())
    {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnPoolingForward\n";
#endif

        //HGSOS looks like the yDesc is used for the workspace, not the poolingDesc

        miStat = miopenPoolingGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess)
        {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspacePooling[yDesc] = devptr;
            sDescToWorkspacePoolingSize[yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }

    }
    else
    {
        devptr = sDescToWorkspacePooling[yDesc];
        workSpaceSize = sDescToWorkspacePoolingSize[yDesc];
    }

    return miopenTohipdnnStatus(
            miopenPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta,
                    yDesc, y, false, // do_backward,
                    (void *) devptr, workSpaceSize));

}

//=================================!

hipdnnStatus_t hipdnnPoolingBackward(hipdnnHandle_t handle,
        const hipdnnPoolingDescriptor_t poolingDesc, const void *alpha,
        const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;

    int8_t* devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnPoolingBackward\n";
#endif

    //HGSOS it appears that forward and backward pooling can reuse tha same map.

    if (sDescToWorkspacePooling.find(yDesc) != sDescToWorkspacePooling.end())
    {
        //HGSOS looks like the yDesc is used for the workspace, not the poolingDesc

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnPoolingBackward\n";
#endif

        miStat = miopenPoolingGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess)
        {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspacePooling[yDesc] = devptr;
            sDescToWorkspacePoolingSize[yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }

    }
    else
    {
        devptr = sDescToWorkspacePooling[yDesc];
        workSpaceSize = sDescToWorkspacePoolingSize[yDesc];
    }

    return miopenTohipdnnStatus(
            miopenPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc,
                    dy, xDesc, x, beta, dxDesc, dx, devptr)); //HGSOS  //NOTYET no worspace size!  const!!!????

}
//=============================================================================

hipdnnStatus_t hipdnnCreateActivationDescriptor(
        hipdnnActivationDescriptor_t *activationDesc)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreateActivationDescriptor\n";
#endif

    return miopenTohipdnnStatus(
            miopenCreateActivationDescriptor(activationDesc));
}
//=============================================================================

hipdnnStatus_t hipdnnSetActivationDescriptor(
        hipdnnActivationDescriptor_t activationDesc,
        hipdnnActivationMode_t mode, hipdnnNanPropagation_t reluNanOpt,
        double reluCeiling)
{
    miopenActivationMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetActivationDescriptor\n";
#endif

    hipTomiopenActivationMode(mode, &mimode);

    return miopenTohipdnnStatus(
            miopenSetActivationDescriptor(activationDesc, mimode, 0, //Alpha
                    0, //Beta
                    1)); //Power
}

//=============================================================================

//HGSOS may need another function to accommodate the 3 parameters.

hipdnnStatus_t hipdnnGetActivationDescriptor(
        const hipdnnActivationDescriptor_t activationDesc,
        hipdnnActivationMode_t *mode, hipdnnNanPropagation_t *reluNanOpt,
        double* reluCeiling)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "Inside hipdnnGetActivationDescriptor\n";
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET

//HGSOS //NOTYET fix the miopenSetActivationDescriptor api first

    miopenStatus_t
    miopenGetActivationDescriptor(
            const miopenActivationDescriptor_t activDesc,
            miopenActivationMode_t *mode,
            double *activAlpha,
            double *activBeta,
            double *activPower)

    miopenActivationDescriptor_t miActDes;
    hipdnnStatus_t retVal;

    retVal = hipTomiopenActivationMode(activationDesc, &miActDes);

    if( retVal != HIPDNN_STATUS_SUCCESS )
    return retVal;
#endif

}

//=============================================================================

hipdnnStatus_t hipdnnDestroyActivationDescriptor(
        hipdnnActivationDescriptor_t activationDesc)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDestroyActivationDescriptor\n";
#endif

    return miopenTohipdnnStatus(
            miopenDestroyActivationDescriptor(activationDesc));
}
//=================

hipdnnStatus_t hipdnnActivationForward(hipdnnHandle_t handle,
        hipdnnActivationDescriptor_t activationDesc, //HGSOS not const in cudnn
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnActivationForward\n";
#endif

    return miopenTohipdnnStatus(
            miopenActivationForward(handle, activationDesc, alpha, xDesc, x,
                    beta, yDesc, y));
}
//======================

hipdnnStatus_t hipdnnActivationBackward(hipdnnHandle_t handle,
        hipdnnActivationDescriptor_t activationDesc, //HGSOS const missing in cuda
        const void *alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnActivationBackward\n";
#endif

    return miopenTohipdnnStatus(
            miopenActivationBackward(handle, activationDesc, alpha, yDesc, y,
                    dyDesc, dy, xDesc, x, beta, dxDesc, dx));
}
//=============================================================================

hipdnnStatus_t hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreateLRNDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenCreateLRNDescriptor(normDesc));
}
//=============================================================================

hipdnnStatus_t hipdnnSetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
        hipdnnLRNMode_t mode, unsigned lrnN, double lrnAlpha, double lrnBeta,
        double lrnK)
{

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreateLRNDescriptor\n";
#endif

    retVal = hipTomiopenLRNMode(mode, &mimode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return miopenTohipdnnStatus(
            miopenSetLRNDescriptor(normDesc, mimode, lrnN, lrnAlpha, lrnBeta,
                    lrnK));

}

//================

hipdnnStatus_t hipdnnGetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
        hipdnnLRNMode_t *mode, unsigned* lrnN, double* lrnAlpha,
        double* lrnBeta, double* lrnK)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreateLRNDescriptor\n";
#endif

    retVal = miopenTohipdnnStatus(
            miopenGetLRNDescriptor(normDesc, &mimode, lrnN, lrnAlpha, lrnBeta,
                    lrnK));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    return miopenTohipLRNMode(mimode, mode);
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t normDesc)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDestroyLRNDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenDestroyLRNDescriptor(normDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelForward(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;

    int8_t* devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnLRNCrossChannelForward\n";
#endif

    retVal = hipTomiopenLRNMode(lrnMode, &mimode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    if (sDescToWorkspaceLRN.find(yDesc) != sDescToWorkspaceLRN.end())
    {
        //HGSOS looks like the yDesc is used for the workspace, not the hipdnnLRNDescriptor_t

        miStat = miopenLRNGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess)
        {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspaceLRN[yDesc] = devptr;
            sDescToWorkspaceLRNSize[yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }

    }
    else
    {
        devptr = sDescToWorkspaceLRN[yDesc];
        workSpaceSize = sDescToWorkspaceLRNSize[yDesc];
    }

    return miopenTohipdnnStatus(
            miopenLRNForward(handle, normDesc, alpha, xDesc, x, beta, yDesc, y,
                    false, // bool do_backward, //HGSOS
                    devptr));  //HGSOS //NOTYET no workspace size
}

hipdnnStatus_t hipdnnLRNCrossChannelForwardEx(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y,
        size_t workspaceSize, void* workspace)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnLRNCrossChannelForward\n";
#endif

    retVal = hipTomiopenLRNMode(lrnMode, &mimode);
    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    //mimode is otherwise unused.

    return miopenTohipdnnStatus(
            miopenLRNForward(handle, normDesc, alpha, xDesc, x, beta, yDesc, y,
                    false, //bool do_backward, //HGSOS //NOTYET
                    workspace));  //NOTYET  no workspace size!
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelBackward(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx)
{
//    return HIPDNN_STATUS_NOT_SUPPORTED;

    int8_t* devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnLRNCrossChannelBackward\n";
#endif

    retVal = hipTomiopenLRNMode(lrnMode, &mimode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    if (sDescToWorkspaceLRN.find(yDesc) != sDescToWorkspaceLRN.end())
    {
        //HGSOS looks like the yDesc is used for the workspace, not the hipdnnLRNDescriptor_t

        miStat = miopenLRNGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess)
        {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspaceLRN[yDesc] = devptr;
            sDescToWorkspaceLRNSize[yDesc] = workSpaceSize;
        }
        else
        {
            return miopenTohipdnnStatus(miStat);
        }

    }
    else
    {
        devptr = sDescToWorkspaceLRN[yDesc];
        workSpaceSize = sDescToWorkspaceLRNSize[yDesc];
    }

    return hipdnnLRNCrossChannelBackwardEx(handle, normDesc, lrnMode, alpha,
            yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpaceSize,
            devptr);
}

hipdnnStatus_t hipdnnLRNCrossChannelBackwardEx(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx, size_t workspacesize, //HGSOS //NOTYET unused!!!
        void* workspace)
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenLRNMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnLRNCrossChannelBackwardEx\n";
#endif

    retVal = hipTomiopenLRNMode(lrnMode, &mimode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    //mimode is otherwise unused.
    return miopenTohipdnnStatus(
            miopenLRNBackward(handle, normDesc, alpha, yDesc, y, dyDesc, dy,
                    xDesc, x, beta, dxDesc, dx, workspace));
}

//==================================!

hipdnnStatus_t hipdnnDeriveBNTensorDescriptor(
        hipdnnTensorDescriptor_t derivedBnDesc,
        const hipdnnTensorDescriptor_t xDesc, hipdnnBatchNormMode_t mode)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDeriveBNTensorDescriptor\n";
#endif

    return miopenTohipdnnStatus(
            miopenDeriveBNTensorDescriptor(derivedBnDesc, xDesc,
                    hipTomiopenBatchNormMode(mode)));
}

//=============================================================================

hipdnnStatus_t hipdnnBatchNormalizationForwardTraining(hipdnnHandle_t handle,
        hipdnnBatchNormMode_t mode, void *alpha, void *beta,
        const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnTensorDescriptor_t yDesc, void *y,
        const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, void *bnScale,
        void *bnBias, double exponentialAverageFactor, void *resultRunningMean,
        void *resultRunningVariance, double epsilon, void *resultSaveMean,
        void *resultSaveInvVariance)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnBatchNormalizationForwardTraining\n";
#endif

    return miopenTohipdnnStatus(
            miopenBatchNormalizationForwardTraining(handle,
                    hipTomiopenBatchNormMode(mode), alpha, beta, xDesc, x,
                    yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias,
                    exponentialAverageFactor, resultRunningMean,
                    resultRunningVariance, epsilon, resultSaveMean,
                    resultSaveInvVariance));
}
//=============================================================================

hipdnnStatus_t hipdnnnBatchNormalizationForwardInference(hipdnnHandle_t handle,
        hipdnnBatchNormMode_t mode, void *alpha, void *beta,
        const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnTensorDescriptor_t yDesc, void *y,
        const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "hipdnnnBatchNormalizationForwardInference NOT IMPLEMENTED" << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET
    //arguments 10,11,12,13 below are not const in miopen.
    return miopenTohipdnnStatus(
            miopenBatchNormalizationForwardInference( handle,
                    hipTomiopenBatchNormMode(mode),
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
#endif

}
//=============================================================================

hipdnnStatus_t hipdnnBatchNormalizationBackward(hipdnnHandle_t handle,
        hipdnnBatchNormMode_t mode, const void *alphaDataDiff,
        const void *betaDataDiff, const void *alphaParamDiff,
        const void *betaParamDiff, const hipdnnTensorDescriptor_t xDesc,
        const void *x, const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t dxDesc, void *dx,
        const hipdnnTensorDescriptor_t bnScaleBiasDiffDesc, const void *bnScale,
        void *resultBnScaleDiff, void *resultBnBiasDiff, double epsilon,
        const void *savedMean, const void *savedInvVariance)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnBatchNormalizationBackward\n";
#endif

    return miopenTohipdnnStatus(
            miopenBatchNormalizationBackward(handle,
                    hipTomiopenBatchNormMode(mode), alphaDataDiff, betaDataDiff,
                    alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc,
                    dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff,
                    resultBnBiasDiff, epsilon, savedMean, savedInvVariance));
}

hipdnnStatus_t hipdnnSetTensorNdDescriptor(hipdnnTensorDescriptor_t tensorDesc,
        hipdnnDataType_t dataType, int nbDims, const int dimA[],
        const int strideA[])
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;


#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetTensorNdDescriptor\n";
#endif

    retVal = hipTomiopenDataType(dataType, &moDT);
    if (retVal == HIPDNN_STATUS_SUCCESS)
        return miopenTohipdnnStatus(
                miopenSetTensorDescriptor(tensorDesc, moDT, nbDims,
                        const_cast<int*>(dimA), const_cast<int*>(strideA)));
    return retVal;
}

hipdnnStatus_t hipdnnGetTensorNdDescriptor(
        const hipdnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
        hipdnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[])
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnGetTensorNdDescriptor\n";
#endif

    retVal = miopenTohipdnnStatus(
            miopenGetTensorDescriptor(tensorDesc, &moDT, dimA, strideA));
    if (retVal == HIPDNN_STATUS_SUCCESS)
    {
        retVal = miopenTohipDataType(moDT, dataType);
        if (retVal == HIPDNN_STATUS_SUCCESS)
            retVal = miopenTohipdnnStatus(
                    miopenGetTensorDescriptorSize(tensorDesc, nbDims));
    }
    return retVal;
}

hipdnnStatus_t hipdnnSetFilterNdDescriptor(hipdnnFilterDescriptor_t filterDesc,
        hipdnnDataType_t dataType, // image data type
        hipdnnTensorFormat_t format, int nbDims, const int filterDimA[])
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetFilterNdDescriptor\n";
#endif

    retVal = hipTomiopenDataType(dataType, &moDT);
    if (retVal == HIPDNN_STATUS_SUCCESS)
        return miopenTohipdnnStatus(
                miopenSetTensorDescriptor(filterDesc, moDT, nbDims,
                        const_cast<int*>(filterDimA),
                        const_cast<int*>(filterDimA)));
    return retVal;
}

hipdnnStatus_t hipdnnGetFilterNdDescriptor(
        const hipdnnFilterDescriptor_t filterDesc, int nbDimsRequested,
        hipdnnDataType_t *dataType, // image data type
        hipdnnTensorFormat_t *format, int *nbDims, int filterDimA[])
{
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    miopenDataType_t moDT;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnGetFilterNdDescriptor\n";
#endif

    retVal = miopenTohipdnnStatus(
            miopenGetTensorDescriptor(filterDesc, &moDT, filterDimA,
                    filterDimA));

    if (retVal == HIPDNN_STATUS_SUCCESS)
    {
        retVal = miopenTohipDataType(moDT, dataType);
        if (retVal == HIPDNN_STATUS_SUCCESS)
        {
            retVal = miopenTohipdnnStatus(
                    miopenGetTensorDescriptorSize(filterDesc, nbDims));
            *format = HIPDNN_TENSOR_NCHW; //miopen defines only this format
        }
    }
    return retVal;
}

hipdnnStatus_t hipdnnDestroyFilterDescriptor(
        hipdnnFilterDescriptor_t filterDesc)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDestroyFilterDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenDestroyTensorDescriptor(filterDesc));
}

//RNN APIs

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t * rnnDesc)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreateRNNDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenCreateRNNDescriptor(rnnDesc));
}

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc)
{
    return miopenTohipdnnStatus(miopenDestroyRNNDescriptor(rnnDesc));
}

hipdnnStatus_t hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
        const int minibatch, const hipdnnDataType_t dataType,
        hipdnnPersistentRNNPlan_t * plan)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
        hipdnnPersistentRNNPlan_t plan)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v6(hipdnnHandle_t handle,
        hipdnnRNNDescriptor_t rnnDesc, const int hiddenSize,
        const int numLayers,
        hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
        hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
        hipdnnRNNMode_t mode, hipdnnRNNAlgo_t algo, hipdnnDataType_t dataType)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc,
        int hiddenSize, int numLayers,
        hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
        hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
        hipdnnRNNMode_t mode, hipdnnDataType_t dataType)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v5(hipdnnRNNDescriptor_t rnnDesc,
        int hiddenSize, int numLayers, hipdnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
        hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
        hipdnnRNNMode_t mode, hipdnnDataType_t dataType)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNWorkspaceSize(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t *xDesc, size_t *sizeInBytes)
{
    return miopenTohipdnnStatus(
            miopenGetRNNWorkspaceSize(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(xDesc), sizeInBytes));
}

hipdnnStatus_t hipdnnGetRNNTrainingReserveSize(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t *xDesc, size_t *sizeInBytes)
{
    return miopenTohipdnnStatus(
            miopenGetRNNTrainingReserveSize(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(xDesc), sizeInBytes));
}

hipdnnStatus_t hipdnnGetRNNParamsSize(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc,
        const hipdnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
        hipdnnDataType_t dataType)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNLinLayerMatrixParams(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int layer,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const int linLayerID, hipdnnFilterDescriptor_t linLayerMatDesc,
        void ** linLayerMat)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNLinLayerBiasParams(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int layer,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const int linLayerID, hipdnnFilterDescriptor_t linLayerBiasDesc,
        void ** linLayerBias)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnRNNForwardInference(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t * xDesc, const void * x,
        const hipdnnTensorDescriptor_t hxDesc, const void * hx,
        const hipdnnTensorDescriptor_t cxDesc, const void * cx,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const hipdnnTensorDescriptor_t *yDesc, void * y,
        const hipdnnTensorDescriptor_t hyDesc, void * hy,
        const hipdnnTensorDescriptor_t cyDesc, void * cy, void * workspace,
        size_t workSpaceSizeInBytes)
{
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnRNNForwardTraining(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t *xDesc, const void * x,
        const hipdnnTensorDescriptor_t hxDesc, const void * hx,
        const hipdnnTensorDescriptor_t cxDesc, const void * cx,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const hipdnnTensorDescriptor_t *yDesc, void * y,
        const hipdnnTensorDescriptor_t hyDesc, void * hy,
        const hipdnnTensorDescriptor_t cyDesc, void * cy, void * workspace,
        size_t workSpaceSizeInBytes, void * reserveSpace,
        size_t reserveSpaceSizeInBytes)
{
    return miopenTohipdnnStatus(
            miopenRNNForwardTraining(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(xDesc), x,
                    const_cast<hipdnnTensorDescriptor_t>(hxDesc), hx,
                    const_cast<hipdnnTensorDescriptor_t>(cxDesc), cx, wDesc, w,
                    const_cast<hipdnnTensorDescriptor_t*>(yDesc), y,
                    const_cast<hipdnnTensorDescriptor_t>(hyDesc), hy,
                    const_cast<hipdnnTensorDescriptor_t>(cyDesc), cy, workspace,
                    workSpaceSizeInBytes, reserveSpace,
                    reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnRNNBackwardData(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t * yDesc, const void * y,
        const hipdnnTensorDescriptor_t * dyDesc, const void * dy,
        const hipdnnTensorDescriptor_t dhyDesc, const void * dhy,
        const hipdnnTensorDescriptor_t dcyDesc, const void * dcy,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const hipdnnTensorDescriptor_t hxDesc, const void * hx,
        const hipdnnTensorDescriptor_t cxDesc, const void * cx,
        const hipdnnTensorDescriptor_t * dxDesc, void * dx,
        const hipdnnTensorDescriptor_t dhxDesc, void * dhx,
        const hipdnnTensorDescriptor_t dcxDesc, void * dcx, void * workspace,
        size_t workSpaceSizeInBytes, void * reserveSpace,
        size_t reserveSpaceSizeInBytes)
{
    return miopenTohipdnnStatus(
            miopenRNNBackwardData(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(yDesc), y,
                    const_cast<hipdnnTensorDescriptor_t*>(dyDesc), dy,
                    const_cast<hipdnnTensorDescriptor_t>(dhyDesc), dhy,
                    const_cast<hipdnnTensorDescriptor_t>(dcyDesc), dcy, wDesc,
                    w, const_cast<hipdnnTensorDescriptor_t>(hxDesc), hx,
                    const_cast<hipdnnTensorDescriptor_t>(cxDesc), cx,
                    const_cast<hipdnnTensorDescriptor_t*>(dxDesc), dx,
                    const_cast<hipdnnTensorDescriptor_t>(dhxDesc), dhx,
                    const_cast<hipdnnTensorDescriptor_t>(dcxDesc), dcx,
                    workspace, workSpaceSizeInBytes, reserveSpace,
                    reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnRNNBackwardWeights(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t * xDesc, const void * x,
        const hipdnnTensorDescriptor_t hxDesc, const void * hx,
        const hipdnnTensorDescriptor_t * yDesc, const void * y,
        const void * workspace, size_t workSpaceSizeInBytes,
        const hipdnnFilterDescriptor_t dwDesc, void * dw,
        const void * reserveSpace, size_t reserveSpaceSizeInBytes)
{
    return miopenTohipdnnStatus(
            miopenRNNBackwardWeights(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(xDesc), x,
                    const_cast<hipdnnTensorDescriptor_t>(hxDesc), hx,
                    const_cast<hipdnnTensorDescriptor_t*>(yDesc), y, dwDesc, dw,
                    const_cast<void*>(workspace), workSpaceSizeInBytes,
                    reserveSpace, reserveSpaceSizeInBytes));
}

hipdnnStatus_t hipdnnSetPoolingNdDescriptor(
        hipdnnPoolingDescriptor_t poolingDesc, const hipdnnPoolingMode_t mode,
        const hipdnnNanPropagation_t maxpoolingNanOpt, int nbDims,
        const int windowDimA[], const int paddingA[], const int strideA[])
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetPoolingNdDescriptor with nbDims :" << nbDims
            << std::endl;
#endif

    if (nbDims == 2)
    {
        // 2D Pooling
        hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
        int windowHeight = windowDimA[0];
        int windowWidth = windowDimA[1];
        int pad_h = paddingA[0];
        int pad_w = paddingA[1];
        int u = strideA[0];
        int v = strideA[1];
        miopenPoolingMode_t pooling_mode;
        retVal = hipTomiopenPoolingMode(mode, &pooling_mode);
        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;
        return miopenTohipdnnStatus(
                miopenSet2dPoolingDescriptor(poolingDesc, pooling_mode,
                        windowHeight, windowWidth, pad_h, pad_w, u, v));
    }
    else
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "Higher dimensions > 2 Pooling is not supported"
                << std::endl;
#endif

        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
}

const char * hipdnnGetErrorString(hipdnnStatus_t status)
{
    switch (status)
    {
    case HIPDNN_STATUS_SUCCESS:
        return "HIPDNN_STATUS_SUCCESS";
    case HIPDNN_STATUS_NOT_INITIALIZED:
        return "HIPDNN_STATUS_NOT_INITIALIZED";
    case HIPDNN_STATUS_ALLOC_FAILED:
        return "HIPDNN_STATUS_ALLOC_FAILED";
    case HIPDNN_STATUS_BAD_PARAM:
        return "HIPDNN_STATUS_ALLOC_FAILED";
    case HIPDNN_STATUS_INTERNAL_ERROR:
        return "HIPDNN_STATUS_INTERNAL_ERROR";
    case HIPDNN_STATUS_INVALID_VALUE:
        return "HIPDNN_STATUS_INVALID_VALUE";
    case HIPDNN_STATUS_EXECUTION_FAILED:
        return "HIPDNN_STATUS_EXECUTION_FAILED";
    case HIPDNN_STATUS_NOT_SUPPORTED:
        return "HIPDNN_STATUS_NOT_SUPPORTED";
    default:
        return "HIPDNN_STATUS_INTERNAL_ERROR";
    }

}

hipdnnStatus_t hipdnnSetConvolutionNdDescriptor(
        hipdnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
        const int padA[], const int filterStrideA[], const int dilationA[],
        hipdnnConvolutionMode_t mode, hipdnnDataType_t computeType) // convolution data type
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetConvolutionNdDescriptor with arrayLength :" << arrayLength
            << std::endl;
#endif

    if (arrayLength == 2)
    {
        int pad_h, pad_w, u, v;
        pad_h = padA[0];
        pad_w = padA[1];
        u = filterStrideA[0];
        v = filterStrideA[1];
        hipdnnStatus_t retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        retVal = miopenTohipdnnStatus(
                miopenInitConvolutionDescriptor(convDesc, miopenConvolution,
                        pad_h, pad_w, u, v, 1, 1));
        return retVal;
    }
    else
    {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
        std::cout << "Inside hipdnnSetConvolutionNdDescriptor NOT SUPPORTED"
        << std::endl;
#endif
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
}

hipdnnStatus_t hipdnnBatchNormalizationForwardInference(hipdnnHandle_t handle,
        hipdnnBatchNormMode_t mode,
        const void *alpha, // alpha[0] = result blend factor
        const void *beta,  // beta[0] = dest layer blend factor
        const hipdnnTensorDescriptor_t xDesc,
        const void *x,     // NxCxHxW
        const hipdnnTensorDescriptor_t yDesc,
        void *y,     // NxCxHxW
        const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnBatchNormalizationForwardInference: NOT IMPLEMENTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnCreateDropoutDescriptor(
        hipdnnDropoutDescriptor_t * dropoutDesc)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnCreateDropoutDescriptor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc,
        hipdnnHandle_t handle, float dropout, void * states,
        size_t stateSizeInBytes, unsigned long long seed)
{
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnSetDropoutDescriptor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDropoutGetStatesSize(hipdnnHandle_t handle,
        size_t * sizeInBytes)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnDropoutGetStatesSize: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyDropoutDescriptor(
        hipdnnDropoutDescriptor_t dropoutDesc)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnDestroyDropoutDescriptor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnCreateReduceTensorDescriptor(
        hipdnnReduceTensorDescriptor_t *reduceTensorDesc)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnCreateReduceTensorDescriptor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetTensor4dDescriptorEx(
        hipdnnTensorDescriptor_t tensorDesc, hipdnnDataType_t dataType, /* image data type */
        int n, /* number of inputs (batch size) */
        int c, /* number of input feature maps */
        int h, /* height of input section */
        int w, /* width of input section */
        int nStride, int cStride, int hStride, int wStride)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnSetTensor4dDescriptorEx: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetReduceTensorDescriptor(
        hipdnnReduceTensorDescriptor_t reduceTensorDesc,
        hipdnnReduceTensorOp_t reduceTensorOp,
        hipdnnDataType_t reduceTensorCompType,
        hipdnnNanPropagation_t reduceTensorNanOpt,
        hipdnnReduceTensorIndices_t reduceTensorIndices,
        hipdnnIndicesType_t reduceTensorIndicesType)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnSetReduceTensorDescriptor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetReductionWorkspaceSize(hipdnnHandle_t handle,
        const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
        const hipdnnTensorDescriptor_t aDesc,
        const hipdnnTensorDescriptor_t cDesc, size_t *sizeInBytes)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnGetReductionWorkspaceSize: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnReduceTensor(hipdnnHandle_t handle,
        const hipdnnReduceTensorDescriptor_t reduceTensorDesc, void *indices,
        size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes,
        const void *alpha, const hipdnnTensorDescriptor_t aDesc, const void *A,
        const void *beta, const hipdnnTensorDescriptor_t cDesc, void *C)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnReduceTensor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyReduceTensorDescriptor(
        hipdnnReduceTensorDescriptor_t reduceTensorDesc)
{

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "hipdnnDestroyReduceTensorDescriptor: NOT SUPPORTED." << std::endl;
#endif

    return HIPDNN_STATUS_NOT_SUPPORTED;
}
