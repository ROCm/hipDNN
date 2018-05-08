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
#include <exception>
#include <map>
#include <logger.h>

#define HIPDNNFLUSH <<std::flush;
#define PROMOTE_TO_SUPPORTED

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

hipdnnStatus_t miopenTohipdnnStatus(miopenStatus_t cStatus) {
    hipdnnStatus_t retVal;
    switch (cStatus) {
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

hipdnnStatus_t hipTomiopenDataType(hipdnnDataType_t in, miopenDataType_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    switch (in) {
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

hipdnnStatus_t miopenTohipDataType(miopenDataType_t in, hipdnnDataType_t* out) {
    switch (in) {
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
        hipdnnOpTensorOp_t* out) {
    switch (in) {
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
        miopenTensorOp_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in) {
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

hipdnnConvolutionMode_t miopenTohipConvolutionMode(miopenConvolutionMode_t in) {
    if (in == miopenConvolution)
        return HIPDNN_CONVOLUTION;
    /*else if( in == miopenCrossCorrelation )
     return HIPDNN_CROSS_CORRELATION;*/ //TODO: to be added
    return HIPDNN_CONVOLUTION;
}

miopenConvolutionMode_t hipTomiopenConvolutionMode(hipdnnConvolutionMode_t in) {
    if (in == HIPDNN_CONVOLUTION)
        return miopenConvolution;
    /*else if( in == HIPDNN_CROSS_CORRELATION )
     return miopenCrossCorrelation;*/ //TODO: to be added
    return miopenConvolution;
}

//=============================================================================

hipdnnStatus_t hipTomiopenPoolingMode(hipdnnPoolingMode_t in,
        miopenPoolingMode_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
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
        hipdnnPoolingMode_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case miopenPoolingMax:
        *out = HIPDNN_POOLING_MAX;
        break;
    case miopenPoolingAverage:
        *out = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
//HGSOS     *out = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
//HGSOS     *out = HIPDNN_POOLING_MAX_DETERMINISTIC;
    default:

        HIPDNN_OPEN_LOG_M("miopenTohipPoolingMode "<< in << ": NOT SUPPORTED." << std::flush);
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenLRNMode(hipdnnLRNMode_t in, miopenLRNMode_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_LRN_WITHIN_CHANNEL:
        *out = miopenLRNWithinChannel;
        break;
    case HIPDNN_LRN_CROSS_CHANNEL:
        *out = miopenLRNCrossChannel;
        break;
    }
    return retVal;
}

hipdnnStatus_t miopenTohipLRNMode(miopenLRNMode_t in, hipdnnLRNMode_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
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

miopenBatchNormMode_t hipTomiopenBatchNormMode(hipdnnBatchNormMode_t in) {
    if (in == HIPDNN_BATCHNORM_PER_ACTIVATION) {

        HIPDNN_OPEN_LOG_M("HIPDNN_BATCHNORM_PER_ACTIVATION"  << std::flush);
        return miopenBNPerActivation;
    } else if (in == HIPDNN_BATCHNORM_SPATIAL) {

        HIPDNN_OPEN_LOG_M("HIPDNN_BATCHNORM_SPATIAL"  << std ::flush);
        return miopenBNSpatial;
    }

    HIPDNN_OPEN_LOG_E("HIPDNN_BATCHNORM_SPATIAL"  << std ::flush);

//HGSOS need to return error code, those are not the only options!
    return miopenBNPerActivation;
}

//=============================================================================

hipdnnStatus_t miopenTohipActivationMode(miopenActivationMode_t in,
        hipdnnActivationMode_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in) {
    case miopenActivationLOGISTIC:
        *out = HIPDNN_ACTIVATION_SIGMOID;
        break;

    case miopenActivationRELU:
        *out = HIPDNN_ACTIVATION_RELU;
        break;

    case miopenActivationTANH:
        *out = HIPDNN_ACTIVATION_TANH;
        break;

    case miopenActivationPATHTRU:
        *out = HIPDNN_ACTIVATION_PATHTRU;
        break;

    case miopenActivationSOFTRELU:
        *out = HIPDNN_ACTIVATION_SOFTRELU;
        break;

    case miopenActivationABS:
        *out = HIPDNN_ACTIVATION_ABS;
        break;

    case miopenActivationPOWER:
        *out = HIPDNN_ACTIVATION_POWER;
        break;

    default:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

hipdnnStatus_t hipTomiopenActivationMode(hipdnnActivationMode_t in,
        miopenActivationMode_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in) {
    case HIPDNN_ACTIVATION_SIGMOID:


        HIPDNN_OPEN_LOG_M ("HIPDNN_ACTIVATION_SIGMOID"  << std::flush);


        *out = miopenActivationLOGISTIC;
        break;

    case HIPDNN_ACTIVATION_RELU:

        HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_RELU"  << std::flush);

        *out = miopenActivationRELU;
        break;

    case HIPDNN_ACTIVATION_TANH:

        HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_TANH"  << std::flush);

        *out = miopenActivationTANH;
        break;

    case HIPDNN_ACTIVATION_PATHTRU:

        HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_TANH"  << std::flush);

        *out = miopenActivationPATHTRU;
        break;

    case HIPDNN_ACTIVATION_SOFTRELU:

        HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_TANH"  << std::flush);

        *out = miopenActivationSOFTRELU;
        break;

    case HIPDNN_ACTIVATION_ABS:

        HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_TANH"  << std::flush);

        *out = miopenActivationABS;
        break;

    case HIPDNN_ACTIVATION_POWER:

        HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_TANH"  << std::flush);

        *out = miopenActivationPOWER;
        break;

    case HIPDNN_ACTIVATION_ELU:

        HIPDNN_OPEN_LOG_E("HIPDNN_ACTIVATION_TANH"  << std::flush);

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;

    case HIPDNN_ACTIVATION_CLIPPED_RELU:

        HIPDNN_OPEN_LOG_E("HIPDNN_ACTIVATION_TANH"  << std::flush);

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;

    default:

        HIPDNN_OPEN_LOG_E("HIPDNN_ACTIVATION_TANH"  << std::flush);

        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionFwdAlgo(hipdnnConvolutionFwdAlgo_t in,
        miopenConvFwdAlgorithm_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    switch (in) {

    case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_GEMM"  << std::flush);


        *out = miopenConvolutionFwdAlgoGEMM;
        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT"  << std::flush);

        *out = miopenConvolutionFwdAlgoDirect;

        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_FFT:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_FFT"  << std::flush);


        *out = miopenConvolutionFwdAlgoFFT;
        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"  << std::flush);

        *out = miopenConvolutionFwdAlgoWinograd;
        break;

    case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"  << std::flush);


        *out = miopenConvolutionFwdAlgoGEMM;
        break;

    default:


        HIPDNN_OPEN_LOG_E("hipdnnConvolutionFwdAlgo_t: " << in << " NOT SUPPORTED."
         << std::flush);


        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

    return retVal;
}

hipdnnStatus_t miopenTohipConvolutionFwdAlgo(miopenConvFwdAlgorithm_t in,
        hipdnnConvolutionFwdAlgo_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {

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

int ConvolutionFwdAlgoCount() {
    return 4;
}

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t GetConvolutionFwdAlgo(int i) {
    hipdnnConvolutionFwdAlgo_t retVal;
    miopenConvFwdAlgorithm_t mialgo;

    if (i < ConvolutionFwdAlgoCount()) {
        mialgo = (miopenConvFwdAlgorithm_t) i;
    } else {
        //for protection
        mialgo = (miopenConvFwdAlgorithm_t) miopenConvolutionFwdAlgoWinograd;
    }
    miopenTohipConvolutionFwdAlgo(mialgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionBwdFilterAlgo(
        hipdnnConvolutionBwdFilterAlgo_t in,
        miopenConvBwdWeightsAlgorithm_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0" 
                << std::flush);


        *out = miopenConvolutionBwdWeightsAlgoGEMM;
        break;

    case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1"  << std::flush);


        *out = miopenConvolutionBwdWeightsAlgoDirect;

        break;
        /*case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
         case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:*/ //TODO: will be added in future
    default:


        HIPDNN_OPEN_LOG_E("hipdnnConvolutionBwdFilterAlgo_t: " << in
        << " NOT SUPPORTED."  << std::flush);


        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }

    return retVal;
}

hipdnnStatus_t miopenTohipConvolutionBwdFilterAlgo(
        miopenConvBwdWeightsAlgorithm_t in,
        hipdnnConvolutionBwdFilterAlgo_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case miopenConvolutionBwdWeightsAlgoGEMM:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        break;
    case miopenConvolutionBwdWeightsAlgoDirect:
        *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        break;
    }

    return retVal;
}

int ConvolutionBwdFilterAlgoCount() {
    return (int) 2;
}

// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t GetConvolutionBwdFilterAlgo(int i) {
    hipdnnConvolutionBwdFilterAlgo_t retVal;
    miopenConvBwdWeightsAlgorithm_t mialgo;

    if (i < ConvolutionBwdFilterAlgoCount()) {
        mialgo = (miopenConvBwdWeightsAlgorithm_t) i;
    } else {
        //for protection
        mialgo =
                (miopenConvBwdWeightsAlgorithm_t) miopenConvolutionBwdWeightsAlgoGEMM;
    }
    miopenTohipConvolutionBwdFilterAlgo(mialgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionBwdDataAlgo(
        hipdnnConvolutionBwdDataAlgo_t in, miopenConvBwdDataAlgorithm_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0" 
                << std::flush);


        *out = miopenConvolutionBwdDataAlgoGEMM;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1"  << std::flush);


        *out = miopenConvolutionBwdDataAlgoDirect;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD"  << std::flush);


        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
         << std::flush);


        *out = miopenConvolutionBwdDataAlgoWinograd;
        break;

    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT"  << std::flush);


        *out = miopenConvolutionBwdDataAlgoFFT;
        break;

        /*case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
         case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:*/ //TODO: to be added in future
    case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM:


        HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM"
         << std::flush);


        *out = miopenTransposeBwdDataAlgoGEMM;
        break;

    default:


        HIPDNN_OPEN_LOG_E("hipdnnConvolutionBwdDataAlgo_t: " << in
        << " NOT SUPPORTED."  << std::flush);


        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }
    return retVal;
}

hipdnnStatus_t miopenTohipConvolutionBwdDataAlgo(
        miopenConvBwdDataAlgorithm_t in, hipdnnConvolutionBwdDataAlgo_t* out) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
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

int ConvolutionBwdDataAlgoCount() {
    return (int) 2;
}

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t GetConvolutionBwdDataAlgo(int i) {
    hipdnnConvolutionBwdDataAlgo_t retVal;
    miopenConvBwdDataAlgorithm_t mialgo;

    if (i < ConvolutionBwdDataAlgoCount()) {
        mialgo = (miopenConvBwdDataAlgorithm_t) i;
    } else {
        //for protection
        mialgo =
                (miopenConvBwdDataAlgorithm_t) miopenConvolutionBwdDataAlgoWinograd;
    }
    miopenTohipConvolutionBwdDataAlgo(mialgo, &retVal);

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipSoftmaxModeSupported(hipdnnSoftmaxMode_t in) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    //HGSOS
    case HIPDNN_SOFTMAX_MODE_INSTANCE:


        HIPDNN_OPEN_LOG_E("HIPDNN_SOFTMAX_MODE_INSTANCE NOT SUPPORTED." 
                << std::flush);


        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_SOFTMAX_MODE_CHANNEL:
        break;
    }
    return retVal;
}

hipdnnStatus_t SoftmaxAlgorithmSupported(hipdnnSoftmaxAlgorithm_t in) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
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
hipdnnStatus_t hipTensorFormatSupported(hipdnnTensorFormat_t in) {
    if (in == HIPDNN_TENSOR_NCHW) {
        HIPDNN_OPEN_LOG_M("HIPDNN_TENSOR_NCHW"  << std::flush);

        return HIPDNN_STATUS_SUCCESS;
    }


    HIPDNN_OPEN_LOG_E("hipdnnTensorFormat_t " << in << " NOT SUPPORTED."
     << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t ConvolutionFwdPreferenceSupported(
        hipdnnConvolutionFwdPreference_t in) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST:
        break;
    case HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }


    if (retVal == HIPDNN_STATUS_NOT_SUPPORTED) {
        HIPDNN_OPEN_LOG_E("hipdnnConvolutionFwdPreference_t " << in
                << " NOT SUPPORTED." 
                << std::flush);
    }


    return retVal;
}

hipdnnStatus_t ConvolutionBwdFilterPreferenceSupported(
        hipdnnConvolutionBwdFilterPreference_t in) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;
    switch (in) {
    case HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:
        break;
    case HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT:
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        break;
    }


    if (retVal == HIPDNN_STATUS_NOT_SUPPORTED) {
        HIPDNN_OPEN_LOG_E("hipdnnConvolutionBwdFilterPreference_t " << in
                << " NOT SUPPORTED."  << std::flush);
    }


    return retVal;
}

//=============================================================================

HIPDNN_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle) {
    sConvolutionForwardAlgorithmWorkspace = 0;
    sConvolutionBackwardDataAlgorithmWorkspace = 0;
    sConvolutionBackwardFilterAlgorithmWorkspace = 0;
    return miopenTohipdnnStatus(miopenCreate(handle));
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle) {
    if (sConvolutionForwardAlgorithmWorkspace != 0) {
        hipFree(sConvolutionForwardAlgorithmWorkspace);
        sConvolutionForwardAlgorithmWorkspace = 0;
    }
    if (sConvolutionBackwardDataAlgorithmWorkspace != 0) {
        hipFree(sConvolutionBackwardDataAlgorithmWorkspace);
        sConvolutionBackwardDataAlgorithmWorkspace = 0;
    }
    if (sConvolutionBackwardFilterAlgorithmWorkspace != 0) {
        hipFree(sConvolutionBackwardFilterAlgorithmWorkspace);
        sConvolutionBackwardFilterAlgorithmWorkspace = 0;
    }

    return miopenTohipdnnStatus(miopenDestroy(handle));
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId) {
    return miopenTohipdnnStatus(miopenSetStream(handle, streamId));
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle,
        hipdnnStream_t *streamId) {
    return miopenTohipdnnStatus(miopenGetStream(handle, streamId));
}

size_t hipdnnGetVersion() {
    return 6000;
}

hipdnnStatus_t hipdnnCreateTensorDescriptor(
        hipdnnTensorDescriptor_t *tensorDesc) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    retVal = miopenTohipdnnStatus(miopenCreateTensorDescriptor(tensorDesc));


        HIPDNN_OPEN_LOG_C("hipdnnCreateTensorDescriptor:" << retVal << " for " << *tensorDesc
                 << std::flush);


    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnSetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
        hipdnnTensorFormat_t format, hipdnnDataType_t dataType, int n, int c,
        int h, int w) {

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
        int *nStride, int *cStride, int *hStride, int *wStride) {
    miopenDataType_t midT;
    hipdnnStatus_t retVal;

    retVal = miopenTohipdnnStatus(
            miopenGet4dTensorDescriptor(tensorDesc, &midT, n, c, h, w, nStride,
                    cStride, hStride, wStride));

    if (retVal != HIPDNN_STATUS_SUCCESS) {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnGetTensor4dDescriptor:" << retVal
                 << std::flush);


        return retVal;
    } else {
        retVal = miopenTohipDataType(midT, dataType);

        if (retVal != HIPDNN_STATUS_SUCCESS) {

            HIPDNN_OPEN_LOG_E("ERROR: hipdnnGetTensor4dDescriptor does not support:" << dataType  << std::flush);

        }
    }
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyTensorDescriptor(
        hipdnnTensorDescriptor_t tensorDesc) {
    return miopenTohipdnnStatus(miopenDestroyTensorDescriptor(tensorDesc));
}

//=============================================================================

//dstValue = alpha[0]*srcValue + beta[0]*priorDstValue

hipdnnStatus_t hipdnnAddTensor(hipdnnHandle_t handle, const void *alpha,
        const hipdnnTensorDescriptor_t aDesc, const void *A, const void *beta,
        const hipdnnTensorDescriptor_t cDesc, void *C) {
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
        const hipdnnTensorDescriptor_t cDesc, void *C) {

    return miopenTohipdnnStatus(
            miopenOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2,
                    bDesc, B, beta, cDesc, C));
}
//=============================================================================

hipdnnStatus_t hipdnnSetTensor(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t yDesc, void *y, const void *valuePtr) {
    return miopenTohipdnnStatus(miopenSetTensor(handle, yDesc, y, valuePtr));
}

//=============================================================================

hipdnnStatus_t hipdnnScaleTensor(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t yDesc, void *y, const void *alpha) {
    return miopenTohipdnnStatus(miopenScaleTensor(handle, yDesc, y, alpha));
}

//=============================================================================

hipdnnStatus_t hipdnnCreateFilterDescriptor(
        hipdnnFilterDescriptor_t *filterDesc) {
    hipdnnStatus_t retVal;

    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateFilterDescriptor, " << filterDesc
             << std::flush);


    //in miopen a filter descriptor is just a typedef to a tensor descriptor
    retVal = hipdnnCreateTensorDescriptor(filterDesc);


    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateFilterDescriptor, " << filterDesc
             << std::flush);


    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnCreateConvolutionDescriptor(
        hipdnnConvolutionDescriptor_t *convDesc) {
    return miopenTohipdnnStatus(miopenCreateConvolutionDescriptor(convDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnSetConvolutionMathType(
        hipdnnConvolutionDescriptor_t convDesc, hipdnnMathType_t mathType) {

    HIPDNN_OPEN_LOG_E("hipdnnSetConvolutionMathType " << mathType
            << " NOT SUPPORTED." 
            << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

//=============================================================================

hipdnnStatus_t hipdnnSetConvolution2dDescriptor(
        hipdnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u,
        int v, int upscalex, int upscaley, hipdnnConvolutionMode_t mode,
        hipdnnDataType_t computeType) {

    return miopenTohipdnnStatus(
            miopenInitConvolutionDescriptor(convDesc,
                    hipTomiopenConvolutionMode(mode), pad_h, pad_w, u, v,
                    upscalex, upscaley));
}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolution2dDescriptor(
        const hipdnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_y,
        int *u, int *v, int *upscalex, int *upscaley,
        hipdnnConvolutionMode_t *mode, hipdnnDataType_t *computeType) {

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
        int *w) {


    HIPDNN_OPEN_LOG_C("HIPDNN_SOFTMAX_MODE_INSTANCE NOT SUPPORTED." 
            << std::flush);


    return miopenTohipdnnStatus(miopenGetConvolutionForwardOutputDim(convDesc, //HGSOSOS should be const in miopen.
            inputTensorDesc, filterDesc, n, c, h, w));
}

//==============================================================================

hipdnnStatus_t hipdnnDestroyConvolutionDescriptor(
        hipdnnConvolutionDescriptor_t convDesc) {
    return miopenTohipdnnStatus(miopenDestroyConvolutionDescriptor(convDesc));
}

//===============================================================================

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithm(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
        int *returnedAlgoCount, hipdnnConvolutionFwdAlgoPerf_t *perfResults) {


    HIPDNN_OPEN_LOG_E("hipdnnFindConvolutionForwardAlgorithm NOT IMPLEMENTED."
             << std::flush);


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
        hipdnnConvolutionFwdAlgo_t *algo) {


    HIPDNN_OPEN_LOG_E("hipdnnGetConvolutionForwardAlgorithm NOT IMPLEMENTED."
             << std::flush);


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


    HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC hipdnnGetConvolutionForwardAlgorithm");


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
        size_t workSpaceSizeInBytes) {


    HIPDNN_OPEN_LOG_C("ENTER hipdnnFindConvolutionForwardAlgorithmEx: WS PTR"
            << workSpace << ", " << workSpaceSizeInBytes 
            << std::flush);


    assert(x);
    assert(w);
    assert(y);

    hipdnnStatus_t retVal = HIPDNN_STATUS_EXECUTION_FAILED;

    miopenConvAlgoPerf_t* miopenPerfResults =
            new miopenConvAlgoPerf_t[requestedAlgoCount];

    if (workSpace == NULL || workSpaceSizeInBytes == 0) {
        size_t size;
        retVal = miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                        convDesc, yDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS) {
            return retVal;
        }

        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: hipdnnFindConvolutionForwardAlgorithmEx size "
        << size << " requested AlgoCount: "
        << requestedAlgoCount  << std::flush);

        hipMalloc((void**) &sConvolutionForwardAlgorithmWorkspace, size);


        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: sConvolutionForwardAlgorithmWorkspace "
        << "WSP = " << sConvolutionForwardAlgorithmWorkspace
        << " size = " << size  << std::flush);



        HIPDNN_OPEN_LOG_C("Size of miopenPerfResults " << sizeof(miopenPerfResults)
         << std::flush);


        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc,
                        w, convDesc, yDesc, y, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults,
                        sConvolutionForwardAlgorithmWorkspace, size, true //exhaustiveSearch
                        ));

        if (retVal != HIPDNN_STATUS_SUCCESS) {
            return retVal;
        }


        HIPDNN_OPEN_LOG_C("HIPDNN EXIT FindConvolutionForwardAlgorithmEx: " << retVal  << std::flush);


    
    } else {

        HIPDNN_OPEN_LOG_I("PREALLOCATED hipdnnFindConvolutionForwardAlgorithmEx size "
        << workSpaceSizeInBytes << ", WS PTR =" << workSpace <<",requested AlgoCount: "
        << requestedAlgoCount  << std::flush);

        hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc,
                        w, convDesc, yDesc, y, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults, workSpace,
                        workSpaceSizeInBytes, true //exhaustiveSearch
                        ));

        miopenTohipConvolutionFwdAlgo(miopenPerfResults->fwd_algo,
                &(perfResults->algo));


        HIPDNN_OPEN_LOG_C("HIPDNN EXIT FindConvolutionForwardAlgorithmEx: retval=" << retVal << ", WS size=" << workSpaceSizeInBytes << ", algo =" << perfResults->algo  << std::flush);


    }

    if (retVal == HIPDNN_STATUS_SUCCESS) {
        for (int i = 0; i < *returnedAlgoCount; i++) {
            retVal = miopenTohipConvolutionFwdAlgo(
                    miopenPerfResults[i].fwd_algo, &(perfResults[i].algo));
            if (retVal != HIPDNN_STATUS_SUCCESS) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
                std::cout << "...failed miopenTohipConvolutionFwdDataAlgo"
                        << std::endl HIPDNNFLUSH;
#endif
            } else {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
                std::cout << "...miopenTohipConvolutionFwdDataAlgo OK"
                        << std::endl HIPDNNFLUSH;
                perfResults[i].status = HIPDNN_STATUS_SUCCESS; //TODO: miopen doesn't contain a 'status' member variable , setting it to success as of now.
                perfResults[i].time = miopenPerfResults[i].time;
                perfResults[i].memory = miopenPerfResults[i].memory;
#endif
            }
        }
    }

    delete[] miopenPerfResults;
    return retVal;
}

//=========================================!

hipdnnStatus_t hipdnnGetConvolutionForwardWorkspaceSize(hipdnnHandle_t handle,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t yDesc, hipdnnConvolutionFwdAlgo_t algo,
        size_t *sizeInBytes) {
    *sizeInBytes = 0;


    HIPDNN_OPEN_LOG_C("HIPDNN ENTER hipdnnGetConvolutionForwardWorkspaceSize, algo ="
            << algo  << std::flush);


    miopenConvFwdAlgorithm_t mialgo;
    hipdnnStatus_t retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);

    if (retVal == HIPDNN_STATUS_SUCCESS) {
        return retVal;

        //in miopen, workspace size does not depend on algo.
        retVal = miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                        convDesc, yDesc, sizeInBytes));
    }


    HIPDNN_OPEN_LOG_C("EXIT hipdnnGetConvolutionForwardWorkspaceSize, retVal = "
            << retVal << ",size = " << *sizeInBytes  << std::flush);


    return retVal;

}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionForward(hipdnnHandle_t handle,
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnFilterDescriptor_t wDesc, const void *w,
        const hipdnnConvolutionDescriptor_t convDesc,
        hipdnnConvolutionFwdAlgo_t algo, void *workSpace,
        size_t workSpaceSizeInBytes, const void *beta,
        const hipdnnTensorDescriptor_t yDesc, void *y) {

    HIPDNN_OPEN_LOG_C("calling hipdnnConvolutionForward."  << std::flush);

    if (workSpace == NULL || workSpaceSizeInBytes == 0) {
        // Allocate sConvolutionForwardAlgorithmWorkspace to gather work space value
        size_t size;
        hipdnnStatus_t retVal;


        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: hipdnnConvolutionForward size."
         << std::flush);


        retVal = miopenTohipdnnStatus(
                miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                        convDesc, yDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

        hipMalloc((void**) &sConvolutionForwardAlgorithmWorkspace, size);


        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: sConvolutionForwardAlgorithmWorkspace "
        << "WSP= " << sConvolutionForwardAlgorithmWorkspace
        << " size = " << size  << std::flush);



        HIPDNN_OPEN_LOG_C("Invoking hipToMopenConvolutionFwdAlgo"  << std::flush);
        HIPDNN_OPEN_LOG_C("Passed algo" << algo  << std::flush);

        miopenConvFwdAlgorithm_t mialgo;
        retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);


        HIPDNN_OPEN_LOG_C("Invoked hipToMopenConvolutionFwdAlgo"  << std::flush);


        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;


        HIPDNN_OPEN_LOG_C("Invoking MiopenConvolutionFwd"  << std::flush);


        return miopenTohipdnnStatus(
                miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w,
                        convDesc, mialgo, beta, yDesc, y,
                        sConvolutionForwardAlgorithmWorkspace, size));

    } else {

        miopenConvFwdAlgorithm_t mialgo;


        HIPDNN_OPEN_LOG_C("Invoking hipToMopenConvolutionFwdAlgo"  << std::flush);
        HIPDNN_OPEN_LOG_C("Passed algo" << algo  << std::flush);


        hipdnnStatus_t retVal = hipTomiopenConvolutionFwdAlgo(algo, &mialgo);


        HIPDNN_OPEN_LOG_C("Invoked hipToMopenConvolutionFwdAlgo"  << std::flush);


        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;


        HIPDNN_OPEN_LOG_C("Invoking MiopenConvolutionFwd"  << std::flush);


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
        void *db) {


    HIPDNN_OPEN_LOG_C("calling hipdnnConvolutionBackwardBias." 
            << std::flush);


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
        hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults) {

    HIPDNN_OPEN_LOG_E("hipdnnFindConvolutionBackwardFilterAlgorithm NOT IMPLEMENTED"
            
            << std::flush);


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
        size_t memoryLimitInBytes, hipdnnConvolutionBwdFilterAlgo_t *algo) {


    HIPDNN_OPEN_LOG_E("hipdnnGetConvolutionBackwardFilterAlgorithm NOT IMPLEMENTED"
            
            << std::flush);


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
        size_t workSpaceSizeInBytes) {


    HIPDNN_OPEN_LOG_C("Inside hipdnnFindConvolutionBackwardFilterAlgorithmEx");


    assert(x);
    assert(dy);
    assert(dw);

    miopenConvAlgoPerf_t* miopenPerfResults =
            new miopenConvAlgoPerf_t[requestedAlgoCount];
    hipdnnStatus_t retVal = HIPDNN_STATUS_EXECUTION_FAILED;

    try {
        if (workSpace == NULL || workSpaceSizeInBytes == 0) {



       HIPDNN_OPEN_LOG_I( "INTERNAL_ALLOC hipdnnFindConvolutionBackwardFilterAlgorithmEx");


            size_t size;
            retVal = miopenTohipdnnStatus(
                    miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle,
                            dyDesc, xDesc, convDesc, dwDesc, &size));
            if (retVal != HIPDNN_STATUS_SUCCESS)
                return retVal;


        HIPDNN_OPEN_LOG_C("miopenConvolutionBackwardGetWorkSpaceSize size " << size
                << " requested AlgoCount: " << requestedAlgoCount 
                << std::flush);


            hipMalloc((void**) &sConvolutionBackwardFilterAlgorithmWorkspace,
                    size);

        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: sConvolutionBackwardFilterAlgorithmWorkspace "
                << "WSP= " << sConvolutionBackwardFilterAlgorithmWorkspace
                << " size = " << size  << std::flush);


            retVal = miopenTohipdnnStatus(
                    miopenFindConvolutionBackwardWeightsAlgorithm(handle,
                            dyDesc, dy, xDesc, x, convDesc, dwDesc, dw,
                            requestedAlgoCount, returnedAlgoCount,
                            miopenPerfResults,
                            sConvolutionBackwardFilterAlgorithmWorkspace, size,
                            true //exhaustiveSearch
                            ));

        } else {

            retVal = miopenTohipdnnStatus(
                    miopenFindConvolutionBackwardWeightsAlgorithm(handle,
                            dyDesc, dy, xDesc, x, convDesc, dwDesc, dw,
                            requestedAlgoCount, returnedAlgoCount,
                            miopenPerfResults, workSpace, workSpaceSizeInBytes,
                            true //exhaustiveSearch
                            ));
        }
    } catch (std::exception& e) {
        std::cout << "EXCEPTION: hipdnnFindConvolutionBackwardFilterAlgorithmEx"
                << e.what() << std::endl HIPDNNFLUSH
    }

    if (retVal == HIPDNN_STATUS_SUCCESS) {
        for (int i = 0; i < *returnedAlgoCount; i++) {
            retVal = miopenTohipConvolutionBwdFilterAlgo(
                    miopenPerfResults[i].bwd_weights_algo,
                    &(perfResults[i].algo));
            if (retVal != HIPDNN_STATUS_SUCCESS) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
                std::cout << "...failed miopenTohipConvolutionBwdFilterAlgo"
                        << std::endl HIPDNNFLUSH;
#endif
            } else {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
                std::cout << "...miopenTohipConvolutionBwdFilterAlgo OK"
                        << std::endl HIPDNNFLUSH;
                perfResults[i].status = HIPDNN_STATUS_SUCCESS; //TODO: miopen doesn't contain a 'status' member variable , setting it to success as of now.
                perfResults[i].time = miopenPerfResults[i].time;
                perfResults[i].memory = miopenPerfResults[i].memory;
#endif
            }
        }
    }

    delete[] miopenPerfResults;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "EXIT: hipdnnFindConvolutionBackwardFilterAlgorithmEx\n";
#endif

    return retVal;
}

//=================HGSOS======================!

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterWorkspaceSize(
        hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnFilterDescriptor_t dwDesc,
        hipdnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
    *sizeInBytes = 0;


    HIPDNN_OPEN_LOG_C("ENTER hipdnnGetConvolutionBackwardFilterWorkspaceSize algo:"
            << algo 
            << std::flush);

    hipdnnStatus_t retVal = HIPDNN_STATUS_SUCCESS;

    retVal = miopenTohipdnnStatus(
            miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc,
                    xDesc, convDesc, dwDesc, sizeInBytes));

    if (retVal != HIPDNN_STATUS_SUCCESS) {

        HIPDNN_OPEN_LOG_E("ERROR hipdnnGetConvolutionBackwardFilterWorkspaceSize."
                
                << std::flush);

    }


    HIPDNN_OPEN_LOG_C("EXIT hipdnnGetConvolutionBackwardFilterWorkspaceSize:"
            << *sizeInBytes  << std::flush);


    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardFilter(hipdnnHandle_t handle,
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnConvolutionDescriptor_t convDesc,
        hipdnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
        size_t workSpaceSizeInBytes, const void *beta,
        const hipdnnFilterDescriptor_t dwDesc, void *dw) {


    HIPDNN_OPEN_LOG_C("CALL_STACK: Inside hipdnnConvolutionBackwardFilter");

    hipdnnStatus_t retVal;

    if (workSpaceSizeInBytes == 0 || workSpace == NULL) {


        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: hipdnnConvolutionBackwardFilter");


        size_t size;
        retVal = miopenTohipdnnStatus(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc,
                        xDesc, convDesc, dwDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS)
            return retVal;

        hipMalloc((void**) &sConvolutionBackwardFilterAlgorithmWorkspace, size);


        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: sConvolutionBackwardFilterAlgorithmWorkspace "
                << "WSP= " << sConvolutionBackwardFilterAlgorithmWorkspace
                << " size = " << size  << std::flush);


        miopenConvBwdWeightsAlgorithm_t mialgo;
        retVal = hipTomiopenConvolutionBwdFilterAlgo(algo, &mialgo);

        return miopenTohipdnnStatus(
                miopenConvolutionBackwardWeights(handle, alpha, dyDesc, dy,
                        xDesc, x, convDesc, mialgo, beta, dwDesc, dw,
                        sConvolutionBackwardFilterAlgorithmWorkspace, size));

    } else {

        HIPDNN_OPEN_LOG_I("PREALLCOATED: hipdnnConvolutionBackwardFilter:" << workSpace
                  << ", size= " << workSpaceSizeInBytes
                  << ",x PTR = "  << x  << std::flush);


        miopenConvBwdWeightsAlgorithm_t mialgo;
        hipdnnStatus_t retVal = hipTomiopenConvolutionBwdFilterAlgo(algo,
                &mialgo);

        if (retVal != HIPDNN_STATUS_SUCCESS) {

    HIPDNN_OPEN_LOG_E("Could not get algo for " << algo  << std::flush);


            return retVal;
        } else {
            retVal = miopenTohipdnnStatus(
                    miopenConvolutionBackwardWeights(handle, alpha, dyDesc, dy,
                            xDesc, x, convDesc, mialgo, beta, dwDesc, dw,
                            workSpace, workSpaceSizeInBytes));


    HIPDNN_OPEN_LOG_C("miopenConvolutionBackwardWeights "
            << "retVal=" << retVal
            << ",handle= " <<  handle
            << ",alpha=" << alpha
            << ",xDesc=" << xDesc
            << ",x=" << x
            << ",dyDesc=" <<  dyDesc
            << ",dy=" << dy
            << ",convDesc=" << convDesc
            << ",algo=" << algo
            << ",workSpace=" << workSpace
            << ",workSpaceSizeInBytes = " << workSpaceSizeInBytes
            << ",beta=" << beta
            << ",dwDesc=" << dwDesc
            << ",dw=" << dw  << std::flush);


        }
        return retVal;
    }

}

//=============================================================================

hipdnnStatus_t hipdnnGetConvolutionBackwardDataWorkspaceSize(
        hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc,
        hipdnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes) {
    *sizeInBytes = 0;
    //does not depend on algo in miopen
    try {
        return miopenTohipdnnStatus(
                miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc,
                        wDesc, convDesc, dxDesc, sizeInBytes));
    } catch (std::exception& e) {
        std::cout
                << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
                << e.what() << std::endl HIPDNNFLUSH;
    }
}

//=============================================================================

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithm(hipdnnHandle_t handle,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
        int *returnedAlgoCount,
        hipdnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    try {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnFindConvolutionBackwardDataAlgorithm NOT IMPLEMENTED"
                 << std::flush);


        return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif
    } catch (std::exception& e) {
        std::cout
                << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
                << e.what() << std::endl HIPDNNFLUSH;
    }
}

hipdnnStatus_t hipdnnGetConvolutionBackwardDataAlgorithm(hipdnnHandle_t handle,
        const hipdnnFilterDescriptor_t wDesc,
        const hipdnnTensorDescriptor_t dyDesc,
        const hipdnnConvolutionDescriptor_t convDesc,
        const hipdnnTensorDescriptor_t dxDesc,
        hipdnnConvolutionBwdDataPreference_t preference,
        size_t memoryLimitInBytes, hipdnnConvolutionBwdDataAlgo_t *algo) {
    try {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnGetConvolutionBackwardDataAlgorithm NOT IMPLEMENTED"
                << std::flush);


        return HIPDNN_STATUS_NOT_SUPPORTED;

#ifdef NOTYET
//HGSOS   Could use the extended version, but don't know how to get x from xDesc etc.
#endif
    } catch (std::exception& e) {
        std::cout
                << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
                << e.what() << std::endl HIPDNNFLUSH;
    }
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


    HIPDNN_OPEN_LOG_C("Inside hipdnnFindConvolutionBackwardDataAlgorithmEx: input ws size="
            << workSpaceSizeInBytes << ", requestedAlgoCount="
            << requestedAlgoCount << ", WS PTR=" << workSpace 
            << std::flush);


    size_t size;
    hipdnnStatus_t retVal;

    miopenConvAlgoPerf_t* miopenPerfResults =
            new miopenConvAlgoPerf_t[requestedAlgoCount];

    try {
        retVal = miopenTohipdnnStatus(
                miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc,
                        wDesc, convDesc, dxDesc, &size));

        if (retVal != HIPDNN_STATUS_SUCCESS) {



            HIPDNN_OPEN_LOG_E( "...failed miopenConvolutionBackwardDataGetWorkSpaceSize"
             << std::flush);


            return retVal;
        }


        if (size == 0)
        {
            HIPDNN_OPEN_LOG_E("...zero WS size"  << std::flush);
        }



        HIPDNN_OPEN_LOG_I( "INTERNAL_ALLOC: miopenConvolutionBackwardGetWorkSpaceSize size "
        << size << " requested AlgoCount: "
        << requestedAlgoCount  << std::flush);


        hipMalloc((void**) &sConvolutionBackwardDataAlgorithmWorkspace, size);


        HIPDNN_OPEN_LOG_I( "INTERNAL_ALLOC: miopenConvolutionBackwardGetWorkSpaceSize "
        << "WSP= " << sConvolutionBackwardDataAlgorithmWorkspace
        << " size = " << size  << std::flush);


        retVal = miopenTohipdnnStatus(
                miopenFindConvolutionBackwardDataAlgorithm(handle, dyDesc, dy,
                        wDesc, w, convDesc, dxDesc, dx, requestedAlgoCount,
                        returnedAlgoCount, miopenPerfResults,
                        sConvolutionBackwardDataAlgorithmWorkspace, size, true // exhaustiveSearch
                        ));

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {

            HIPDNN_OPEN_LOG_E("...failed miopenFindConvolutionBackwardDataAlgorithm, returnedAlgoCount:"
            << *returnedAlgoCount  << std::flush);

        }
        else
        {

            HIPDNN_OPEN_LOG_C( "...miopenFindConvolutionBackwardDataAlgorithm OK, returnedAlgoCount:"
            << *returnedAlgoCount  << std::flush);

        }

//HGSOS    workSpace = sConvolutionBackwardDataAlgorithmWorkspace;
    } catch (std::exception& e) {
        std::cout
                << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
                << e.what() << std::endl HIPDNNFLUSH;
    }

    if (retVal == HIPDNN_STATUS_SUCCESS) {
        for (int i = 0; i < *returnedAlgoCount; i++) {
            retVal = miopenTohipConvolutionBwdDataAlgo(
                    miopenPerfResults[i].bwd_data_algo, &(perfResults[i].algo));

            if (retVal != HIPDNN_STATUS_SUCCESS) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
                std::cout << "...failed miopenTohipConvolutionBwdDataAlgo"
                        << std::endl HIPDNNFLUSH;
#endif
            } else {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
                std::cout << "...miopenTohipConvolutionBwdDataAlgo OK"
                        << std::endl HIPDNNFLUSH;
                perfResults[i].status = HIPDNN_STATUS_SUCCESS; //TODO: miopen doesn't contain a 'status' member variable , setting it to success as of now.
                perfResults[i].time = miopenPerfResults[i].time;
                perfResults[i].memory = miopenPerfResults[i].memory;
#endif
            }
        }
    }
    delete[] miopenPerfResults;

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipdnnConvolutionBackwardData(hipdnnHandle_t handle,
        const void *alpha, const hipdnnFilterDescriptor_t wDesc, const void *w,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnConvolutionDescriptor_t convDesc,
        hipdnnConvolutionBwdDataAlgo_t algo, void *workSpace,
        size_t workSpaceSizeInBytes, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx) {


    HIPDNN_OPEN_LOG_C("ConvolutionBackwardData: WS PTR=" << workSpace
            << ", WS size = " << workSpaceSizeInBytes  << std::flush);

    try
    {
        if (workSpace == NULL || workSpaceSizeInBytes == 0)
        {

            HIPDNN_OPEN_LOG_I("ConvolutionBackwardData: INTERNAL_ALLOC: hipdnnConvolutionBackwardData");


            size_t size;
            hipdnnStatus_t retVal;
            retVal = miopenTohipdnnStatus(
                    miopenConvolutionBackwardDataGetWorkSpaceSize(handle,
                            dyDesc, wDesc, convDesc, dxDesc, &size));

            if (retVal != HIPDNN_STATUS_SUCCESS)
            {

                HIPDNN_OPEN_LOG_E( "ConvolutionBackwardData: failed miopenConvolutionBackwardDataGetWorkSpaceSize: "
                << retVal  << std::flush);


                return retVal;
            }

            hipMalloc((void**) &sConvolutionBackwardDataAlgorithmWorkspace,
                    size);


            HIPDNN_OPEN_LOG_I("ConvolutionBackwardData: Allocated workspace "
            << "WSP = "
            << sConvolutionBackwardDataAlgorithmWorkspace
            << ", size:" << size  << std::flush);


            // Allocate sConvolutionBackwardDataAlgorithmWorkspace to gather work space value
            miopenConvBwdDataAlgorithm_t mialgo;
            retVal = hipTomiopenConvolutionBwdDataAlgo(algo, &mialgo);

            if (retVal != HIPDNN_STATUS_SUCCESS)
            {

               HIPDNN_OPEN_LOG_E( "ConvolutionBackwardData: failed hipTomiopenConvolutionBwdDataAlgo: "
                "HIP algo = " << algo
                << ", miopen algo= " << mialgo << ", error = "
                << retVal  << std::flush);


                return retVal;
            }

            HIPDNN_OPEN_LOG_C( "ConvolutionBackwardData:  hipTomiopenConvolutionBwdDataAlgo OK."
             << std::flush);
            HIPDNN_OPEN_LOG_C( "ConvolutionBackwardData: about to invoke miopenConvolutionBackwardData."
            << ", WS PTR = "
            << sConvolutionBackwardDataAlgorithmWorkspace
            << ", WS size =" << size  << std::flush);


            return miopenTohipdnnStatus(
                    miopenConvolutionBackwardData(handle, alpha, dyDesc, dy,
                            wDesc, w, convDesc, mialgo, beta, dxDesc, dx,
                            sConvolutionBackwardDataAlgorithmWorkspace, size));

        }
        else
        {

            HIPDNN_OPEN_LOG_C( "ConvolutionBackwardData: workspace preallocated."
             << std::flush);


            miopenConvBwdDataAlgorithm_t mialgo;
            hipdnnStatus_t retVal = hipTomiopenConvolutionBwdDataAlgo(algo,
                    &mialgo);

            if (retVal != HIPDNN_STATUS_SUCCESS)
            {

               HIPDNN_OPEN_LOG_E( "ConvolutionBackwardData: failed hipTomiopenConvolutionBwdDataAlgo: "
                "HIP algo = " << algo
                << ", miopen algo= " << mialgo << ", error = "
                << retVal  << std::flush);

                return retVal;
            }

            float a = *(static_cast<const float*>(alpha));
            float b = *(static_cast<const float*>(beta));


            HIPDNN_OPEN_LOG_C("ConvolutionBackwardData: alpha and beta values are "
            << a << " and " << b  << std::flush);


            return miopenTohipdnnStatus(
                    miopenConvolutionBackwardData(handle, alpha, dyDesc, dy,
                            wDesc, w, convDesc, mialgo, beta, dxDesc, dx,
                            workSpace, workSpaceSizeInBytes));


            HIPDNN_OPEN_LOG_C( "ConvolutionBackwardData: Invoked miopenConvolutionBackwardData "
             << std::flush);


        }

    } catch (std::exception& e) {
        std::cout
                << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
                << e.what() << std::endl HIPDNNFLUSH;
    }
}

//=============================================================================

hipdnnStatus_t hipdnnSoftmaxForward(hipdnnHandle_t handle,
        hipdnnSoftmaxAlgorithm_t algo, hipdnnSoftmaxMode_t mode,
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y) {

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
        const hipdnnTensorDescriptor_t dxDesc, void *dx) {
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
        hipdnnPoolingDescriptor_t *poolingDesc) {
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
        int verticalStride, int horizontalStride) {

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
        int *verticalStride, int *horizontalStride) {
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
        int *w) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnGetPooling2dDescriptor\n";
#endif

    return miopenTohipdnnStatus(
            miopenGetPoolingForwardOutputDim(poolingDesc, inputTensorDesc, n, c,
                    h, w));
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyPoolingDescriptor(
        hipdnnPoolingDescriptor_t poolingDesc) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDestroyPoolingDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenDestroyPoolingDescriptor(poolingDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingForward(hipdnnHandle_t handle,
        const hipdnnPoolingDescriptor_t poolingDesc, const void *alpha,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t yDesc, void *y) {
//    return HIPDNN_STATUS_NOT_SUPPORTED;
    int8_t* devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnPoolingForward\n";
#endif

    if (sDescToWorkspacePooling.find(yDesc) == sDescToWorkspacePooling.end()) {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnPoolingForward\n";
#endif

        //HGSOS looks like the yDesc is used for the workspace, not the poolingDesc

        miStat = miopenPoolingGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess) {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspacePooling[yDesc] = devptr;
            sDescToWorkspacePoolingSize[yDesc] = workSpaceSize;
        } else {
            return miopenTohipdnnStatus(miStat);
        }

    } else {
        devptr = sDescToWorkspacePooling[yDesc];
        workSpaceSize = sDescToWorkspacePoolingSize[yDesc];
    }

    return miopenTohipdnnStatus(
            miopenPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta,
                    yDesc, y, true, // do_backward,
                    (void *) devptr, workSpaceSize));

}

//=================================!

hipdnnStatus_t hipdnnPoolingBackward(hipdnnHandle_t handle,
        const hipdnnPoolingDescriptor_t poolingDesc, const void *alpha,
        const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx) {

    int8_t* devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnPoolingBackward\n";
#endif

    //HGSOS it appears that forward and backward pooling can reuse tha same map.

    if (sDescToWorkspacePooling.find(yDesc) == sDescToWorkspacePooling.end()) {
        //HGSOS looks like the yDesc is used for the workspace, not the poolingDesc

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC
        std::cout << "INTERNAL_ALLOC: hipdnnPoolingBackward\n";
#endif

        miStat = miopenPoolingGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess) {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspacePooling[yDesc] = devptr;
            sDescToWorkspacePoolingSize[yDesc] = workSpaceSize;
        } else {
            return miopenTohipdnnStatus(miStat);
        }

    } else {
        devptr = sDescToWorkspacePooling[yDesc];
        workSpaceSize = sDescToWorkspacePoolingSize[yDesc];
    }

    return miopenTohipdnnStatus(
            miopenPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc,
                    dy, xDesc, x, beta, dxDesc, dx, devptr)); //HGSOS  //NOTYET no worspace size!  const!!!????

}
//=============================================================================

hipdnnStatus_t hipdnnCreateActivationDescriptor(
        hipdnnActivationDescriptor_t *activationDesc) {
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
        double reluCeilingOrAlpha, double activBeta, double activExp) {
    miopenActivationMode_t mimode;

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnSetActivationDescriptor\n";
#endif

    hipTomiopenActivationMode(mode, &mimode);

    return miopenTohipdnnStatus(
            miopenSetActivationDescriptor(activationDesc, mimode,
                    reluCeilingOrAlpha, activBeta, activExp));
}

//=============================================================================

hipdnnStatus_t hipdnnGetActivationDescriptor(
        const hipdnnActivationDescriptor_t activationDesc,
        hipdnnActivationMode_t *mode, hipdnnNanPropagation_t *reluNanOpt,
        double* reluCeilingOrAlpha, double* activBeta, double* activExp) {

#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_ERRORS
    std::cout << "ENTER hipdnnGetActivationDescriptor\n";
#endif

    hipdnnStatus_t retVal;

    miopenActivationMode_t miactmode;

    retVal = miopenTohipdnnStatus(
            miopenGetActivationDescriptor(
                    (miopenActivationDescriptor_t) activationDesc, &miactmode,
                    reluCeilingOrAlpha, activBeta, activExp));

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    retVal = miopenTohipActivationMode(miactmode, mode);

    if (retVal != HIPDNN_STATUS_SUCCESS)
        return retVal;

    *reluNanOpt = HIPDNN_PROPAGATE_NAN;


    HIPDNN_OPEN_LOG_E("EXIT hipdnnGetActivationDescriptor: " << retVal << std::flush);


    return retVal;

}

//=============================================================================

hipdnnStatus_t hipdnnDestroyActivationDescriptor(
        hipdnnActivationDescriptor_t activationDesc)
{

    HIPDNN_OPEN_LOG_C("Inside hipdnnDestroyActivationDescriptor");


    return miopenTohipdnnStatus(
            miopenDestroyActivationDescriptor(activationDesc));
}
//=================

hipdnnStatus_t hipdnnActivationForward(hipdnnHandle_t handle,
        hipdnnActivationDescriptor_t activationDesc, //HGSOS not const in cudnn
        const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y) {


    HIPDNN_OPEN_LOG_C("Inside hipdnnActivationForward");


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

    HIPDNN_OPEN_LOG_C("Inside hipdnnActivationBackward");


    return miopenTohipdnnStatus(
            miopenActivationBackward(handle, activationDesc, alpha, yDesc, y,
                    dyDesc, dy, xDesc, x, beta, dxDesc, dx));
}
//=============================================================================

hipdnnStatus_t hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnCreateLRNDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenCreateLRNDescriptor(normDesc));
}
//=============================================================================

hipdnnStatus_t hipdnnSetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
        hipdnnLRNMode_t mode, unsigned lrnN, double lrnAlpha, double lrnBeta,
        double lrnK) {

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
        double* lrnBeta, double* lrnK) {
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

hipdnnStatus_t hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t normDesc) {
#if DEBUG_CURRENT_CALL_STACK_LEVEL >= DEBUG_CALL_STACK_LEVEL_CALLS
    std::cout << "Inside hipdnnDestroyLRNDescriptor\n";
#endif

    return miopenTohipdnnStatus(miopenDestroyLRNDescriptor(normDesc));
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelForward(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y) {
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

    if (sDescToWorkspaceLRN.find(yDesc) == sDescToWorkspaceLRN.end()) {
        //HGSOS looks like the yDesc is used for the workspace, not the hipdnnLRNDescriptor_t

        miStat = miopenLRNGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess) {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspaceLRN[yDesc] = devptr;
            sDescToWorkspaceLRNSize[yDesc] = workSpaceSize;
        } else {
            return miopenTohipdnnStatus(miStat);
        }

    } else {
        devptr = sDescToWorkspaceLRN[yDesc];
        workSpaceSize = sDescToWorkspaceLRNSize[yDesc];
    }

    return miopenTohipdnnStatus(
            miopenLRNForward(handle, normDesc, alpha, xDesc, x, beta, yDesc, y,
                    false, // bool do_backward, //HGSOS
                    devptr)); //HGSOS //NOTYET no workspace size
}

hipdnnStatus_t hipdnnLRNCrossChannelForwardEx(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
        const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y,
        size_t workspaceSize, void* workspace) {
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
                    workspace)); //NOTYET  no workspace size!
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelBackward(hipdnnHandle_t handle,
        hipdnnLRNDescriptor_t normDesc, hipdnnLRNMode_t lrnMode,
        const void* alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
        const hipdnnTensorDescriptor_t dyDesc, const void *dy,
        const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
        const hipdnnTensorDescriptor_t dxDesc, void *dx) {
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

    if (sDescToWorkspaceLRN.find(yDesc) == sDescToWorkspaceLRN.end()) {
        //HGSOS looks like the yDesc is used for the workspace, not the hipdnnLRNDescriptor_t

        miStat = miopenLRNGetWorkSpaceSize(yDesc, &workSpaceSize);
        if (miStat == miopenStatusSuccess) {
            hipMalloc((void**) &devptr, workSpaceSize);
            sDescToWorkspaceLRN[yDesc] = devptr;
            sDescToWorkspaceLRNSize[yDesc] = workSpaceSize;
        } else {
            return miopenTohipdnnStatus(miStat);
        }

    } else {
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
        void* workspace) {
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
        const hipdnnTensorDescriptor_t xDesc, hipdnnBatchNormMode_t mode) {

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
        void *resultSaveInvVariance) {

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
        const void *estimatedVariance, double epsilon) {


    HIPDNN_OPEN_LOG_E("hipdnnnBatchNormalizationForwardInference NOT IMPLEMENTED"
            
            << std::flush);


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
        const void *savedMean, const void *savedInvVariance) {


    HIPDNN_OPEN_LOG_C("Inside hipdnnBatchNormalizationBackward");


    return miopenTohipdnnStatus(
            miopenBatchNormalizationBackward(handle,
                    hipTomiopenBatchNormMode(mode), alphaDataDiff, betaDataDiff,
                    alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc,
                    dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff,
                    resultBnBiasDiff, epsilon, savedMean, savedInvVariance));
}

hipdnnStatus_t hipdnnSetTensorNdDescriptor(hipdnnTensorDescriptor_t tensorDesc,
        hipdnnDataType_t dataType, int nbDims, const int dimA[],
        const int strideA[]) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    miopenDataType_t moDT;


    HIPDNN_OPEN_LOG_C("ENTER: hipdnnSetTensorNdDescriptor " << tensorDesc
              << "... nbDims=" << nbDims  << std::flush);


    if (dataType != HIPDNN_DATA_FLOAT)
    {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnSetTensorNdDescriptor only supports floats:"
                << dataType  << std::flush);
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;

    } else {
        retVal = hipTomiopenDataType(dataType, &moDT);
        if (retVal == HIPDNN_STATUS_SUCCESS) {
            retVal = miopenTohipdnnStatus(
                    miopenSetTensorDescriptor(tensorDesc, moDT, nbDims,
                            const_cast<int*>(dimA),
                            const_cast<int*>(strideA)));
        }
        else
        {

            HIPDNN_OPEN_LOG_E("ERROR: hipdnnSetTensorNdDescriptor only supports floats:" << dataType  << std::flush);
            retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        }
    }


    HIPDNN_OPEN_LOG_C("EXIT: hipdnnSetTensorNdDescriptor."  << std::flush);


    return retVal;
}

hipdnnStatus_t hipdnnGetTensorNdDescriptor(
        const hipdnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
        hipdnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    miopenDataType_t moDT;


    HIPDNN_OPEN_LOG_C("ENTER hipdnnGetTensorNdDescriptor " << tensorDesc  << std::flush);


    retVal = miopenTohipdnnStatus(
            miopenGetTensorDescriptor(tensorDesc, &moDT, dimA, strideA));

    if (retVal == HIPDNN_STATUS_SUCCESS) {
        retVal = miopenTohipDataType(moDT, dataType);
        if (retVal == HIPDNN_STATUS_SUCCESS) {
            retVal = miopenTohipdnnStatus(
                    miopenGetTensorDescriptorSize(tensorDesc, nbDims));
        }
        else
        {

            HIPDNN_OPEN_LOG_E("ERROR: hipdnnGetTensorNdDescriptor does not support:"
                    << moDT << ", " << *dataType  << std::flush);

        }
    }
    else
    {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnGetTensorNdDescriptor" << retVal 
                << std::flush);

    }


    HIPDNN_OPEN_LOG_C( "EXIT hipdnnGetTensorNdDescriptor, datatype  (miopen, hipdnn)= "
            << moDT << ", " << *dataType << ",size=" << *nbDims
             << std::flush);

    return retVal;
}

hipdnnStatus_t hipdnnSetFilterNdDescriptor(hipdnnFilterDescriptor_t filterDesc,
        hipdnnDataType_t dataType, // image data type
        hipdnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    miopenDataType_t moDT;


    HIPDNN_OPEN_LOG_C("ENTER hipdnnSetFilterNdDescriptor " << filterDesc  << std::flush);


    retVal = hipTomiopenDataType(dataType, &moDT);
    if (retVal == HIPDNN_STATUS_SUCCESS) {
        retVal = miopenTohipdnnStatus(
                miopenSetTensorDescriptor(filterDesc, moDT, nbDims,
                        const_cast<int*>(filterDimA),
                        const_cast<int*>(filterDimA)));

        if (retVal != HIPDNN_STATUS_SUCCESS)
        {

            HIPDNN_OPEN_LOG_E("ERROR: hipdnnSetFilterNdDescriptor" << retVal
                     << std::flush);
            retVal = HIPDNN_STATUS_NOT_SUPPORTED;
        }
    }
    else
    {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnSetFilterNdDescriptor only supports floats:" << dataType  << std::flush);
        retVal = HIPDNN_STATUS_NOT_SUPPORTED;

    }

    HIPDNN_OPEN_LOG_C("EXIT hipdnnSetFilterNdDescriptor."  << std::flush);

    return retVal;
}

hipdnnStatus_t hipdnnGetFilterNdDescriptor(
        const hipdnnFilterDescriptor_t filterDesc, int nbDimsRequested,
        hipdnnDataType_t *dataType, // image data type
        hipdnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_INTERNAL_ERROR;
    miopenDataType_t moDT;


    HIPDNN_OPEN_LOG_C("ENTER hipdnnGetFilterNdDescriptor " << filterDesc  << std::flush );


    retVal = miopenTohipdnnStatus(
            miopenGetTensorDescriptor(filterDesc, &moDT, filterDimA,
                    filterDimA));

    if (retVal == HIPDNN_STATUS_SUCCESS) {
        retVal = miopenTohipDataType(moDT, dataType);
        if (retVal == HIPDNN_STATUS_SUCCESS) {
            retVal = miopenTohipdnnStatus(
                    miopenGetTensorDescriptorSize(filterDesc, nbDims));
            *format = HIPDNN_TENSOR_NCHW; //miopen defines only this format
        }
    }

    if (retVal != HIPDNN_STATUS_SUCCESS)
    {

        HIPDNN_OPEN_LOG_E("ERROR: hipdnnGetFilterNdDescriptor " << retVal
                 << std::flush);

    }


    HIPDNN_OPEN_LOG_C("EXIT hipdnnGetFilterNdDescriptor");


    return retVal;
}

hipdnnStatus_t hipdnnDestroyFilterDescriptor(
        hipdnnFilterDescriptor_t filterDesc) {
    hipdnnStatus_t retVal = HIPDNN_STATUS_INTERNAL_ERROR;


    HIPDNN_OPEN_LOG_C("ENTER hipdnnDestroyFilterDescriptor " << filterDesc
             << std::flush);


    retVal = miopenTohipdnnStatus(miopenDestroyTensorDescriptor(filterDesc));


    HIPDNN_OPEN_LOG_C("EXIT hipdnnDestroyFilterDescriptor."  << std::flush);


    return retVal;
}

//RNN APIs

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t * rnnDesc) {


    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateRNNDescriptor");


    return miopenTohipdnnStatus(miopenCreateRNNDescriptor(rnnDesc));
}

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc) {
    return miopenTohipdnnStatus(miopenDestroyRNNDescriptor(rnnDesc));
}

hipdnnStatus_t hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
        const int minibatch, const hipdnnDataType_t dataType,
        hipdnnPersistentRNNPlan_t * plan) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
        hipdnnPersistentRNNPlan_t plan) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v6(hipdnnHandle_t handle,
        hipdnnRNNDescriptor_t rnnDesc, const int hiddenSize,
        const int numLayers,
        hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
        hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
        hipdnnRNNMode_t mode, hipdnnRNNAlgo_t algo, hipdnnDataType_t dataType) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc,
        int hiddenSize, int numLayers,
        hipdnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
        hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
        hipdnnRNNMode_t mode, hipdnnDataType_t dataType) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v5(hipdnnRNNDescriptor_t rnnDesc,
        int hiddenSize, int numLayers, hipdnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
        hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
        hipdnnRNNMode_t mode, hipdnnDataType_t dataType) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNWorkspaceSize(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t *xDesc, size_t *sizeInBytes) {
    return miopenTohipdnnStatus(
            miopenGetRNNWorkspaceSize(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(xDesc), sizeInBytes));
}

hipdnnStatus_t hipdnnGetRNNTrainingReserveSize(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int seqLength,
        const hipdnnTensorDescriptor_t *xDesc, size_t *sizeInBytes) {
    return miopenTohipdnnStatus(
            miopenGetRNNTrainingReserveSize(handle, rnnDesc, seqLength,
                    const_cast<hipdnnTensorDescriptor_t*>(xDesc), sizeInBytes));
}

hipdnnStatus_t hipdnnGetRNNParamsSize(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc,
        const hipdnnTensorDescriptor_t xDesc, size_t *sizeInBytes,
        hipdnnDataType_t dataType) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNLinLayerMatrixParams(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int layer,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const int linLayerID, hipdnnFilterDescriptor_t linLayerMatDesc,
        void ** linLayerMat) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNLinLayerBiasParams(hipdnnHandle_t handle,
        const hipdnnRNNDescriptor_t rnnDesc, const int layer,
        const hipdnnTensorDescriptor_t xDesc,
        const hipdnnFilterDescriptor_t wDesc, const void * w,
        const int linLayerID, hipdnnFilterDescriptor_t linLayerBiasDesc,
        void ** linLayerBias) {
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
        size_t workSpaceSizeInBytes) {
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
        size_t reserveSpaceSizeInBytes) {
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
        size_t reserveSpaceSizeInBytes) {
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
        const void * reserveSpace, size_t reserveSpaceSizeInBytes) {
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
        const int windowDimA[], const int paddingA[], const int strideA[]) {


    HIPDNN_OPEN_LOG_C("Inside hipdnnSetPoolingNdDescriptor with nbDims :" << nbDims
            
            << std::flush);


    if (nbDims == 2) {
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
                        windowHeight, windowWidth, pad_h, pad_w,
                        u, v));
    }
    else
    {

        HIPDNN_OPEN_LOG_E("Higher dimensions > 2 Pooling is not supported"
         << std::flush);


        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
}

const char * hipdnnGetErrorString(hipdnnStatus_t status) {
    switch (status) {
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

    HIPDNN_OPEN_LOG_C("Inside hipdnnSetConvolutionNdDescriptor with arrayLength :"
            << arrayLength  << std::flush);


    if (arrayLength == 2) {
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

        HIPDNN_OPEN_LOG_E("Inside hipdnnSetConvolutionNdDescriptor NOT SUPPORTED"
         << std::flush);

        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
}

hipdnnStatus_t hipdnnBatchNormalizationForwardInference(hipdnnHandle_t handle,
        hipdnnBatchNormMode_t mode,
        const void *alpha, // alpha[0] = result blend factor
        const void *beta, // beta[0] = dest layer blend factor
        const hipdnnTensorDescriptor_t xDesc,
        const void *x, // NxCxHxW
        const hipdnnTensorDescriptor_t yDesc,
        void *y, // NxCxHxW
        const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
        const void *bnScale, const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon) {


    HIPDNN_OPEN_LOG_E("hipdnnBatchNormalizationForwardInference: NOT IMPLEMENTED."
            << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnCreateDropoutDescriptor(
        hipdnnDropoutDescriptor_t * dropoutDesc) {


    HIPDNN_OPEN_LOG_E("hipdnnCreateDropoutDescriptor: NOT SUPPORTED." 
            << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc,
        hipdnnHandle_t handle, float dropout, void * states,
        size_t stateSizeInBytes, unsigned long long seed)
{

    HIPDNN_OPEN_LOG_E("hipdnnSetDropoutDescriptor: NOT SUPPORTED." << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDropoutGetStatesSize(hipdnnHandle_t handle,
        size_t * sizeInBytes) {


    HIPDNN_OPEN_LOG_E("hipdnnDropoutGetStatesSize: NOT SUPPORTED." << std::endl
            << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyDropoutDescriptor(
        hipdnnDropoutDescriptor_t dropoutDesc) {


    HIPDNN_OPEN_LOG_E("hipdnnDestroyDropoutDescriptor: NOT SUPPORTED." << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnCreateReduceTensorDescriptor(
        hipdnnReduceTensorDescriptor_t *reduceTensorDesc) {


    HIPDNN_OPEN_LOG_E("hipdnnCreateReduceTensorDescriptor: NOT SUPPORTED."
           << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetTensor4dDescriptorEx(
        hipdnnTensorDescriptor_t tensorDesc, hipdnnDataType_t dataType, /* image data type */
        int n, /* number of inputs (batch size) */
        int c, /* number of input feature maps */
        int h, /* height of input section */
        int w, /* width of input section */
        int nStride, int cStride, int hStride, int wStride) {


    HIPDNN_OPEN_LOG_E("hipdnnSetTensor4dDescriptorEx: NOT SUPPORTED." << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetReduceTensorDescriptor(
        hipdnnReduceTensorDescriptor_t reduceTensorDesc,
        hipdnnReduceTensorOp_t reduceTensorOp,
        hipdnnDataType_t reduceTensorCompType,
        hipdnnNanPropagation_t reduceTensorNanOpt,
        hipdnnReduceTensorIndices_t reduceTensorIndices,
        hipdnnIndicesType_t reduceTensorIndicesType) {


    HIPDNN_OPEN_LOG_E("hipdnnSetReduceTensorDescriptor: NOT SUPPORTED." 
            << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetReductionWorkspaceSize(hipdnnHandle_t handle,
        const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
        const hipdnnTensorDescriptor_t aDesc,
        const hipdnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {


    HIPDNN_OPEN_LOG_E("hipdnnGetReductionWorkspaceSize: NOT SUPPORTED." 
            << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnReduceTensor(hipdnnHandle_t handle,
        const hipdnnReduceTensorDescriptor_t reduceTensorDesc, void *indices,
        size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes,
        const void *alpha, const hipdnnTensorDescriptor_t aDesc, const void *A,
        const void *beta, const hipdnnTensorDescriptor_t cDesc, void *C) {


    HIPDNN_OPEN_LOG_E("hipdnnReduceTensor: NOT SUPPORTED."  << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyReduceTensorDescriptor(
        hipdnnReduceTensorDescriptor_t reduceTensorDesc)
{

    HIPDNN_OPEN_LOG_E("hipdnnDestroyReduceTensorDescriptor: NOT SUPPORTED."
             << std::flush);


    return HIPDNN_STATUS_NOT_SUPPORTED;
}
