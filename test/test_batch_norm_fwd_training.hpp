#ifndef TEST_BATCH_NORM_FWD_TRAINING_HPP
#define TEST_BATCH_NORM_FWD_TRAINING_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_batchnorm_fwd_train(BNorm_params_t &d, dataType *src,
                                dataType *dst, float *avg_time) {

	hipdnnHandle_t hipdnn;
	checkHIPDNN(hipdnnCreate(&hipdnn));
	
	hipdnnTensorDescriptor_t in_desc;
        checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
        checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        d.mb, d.ic, d.ih, d.iw));
		
	hipdnnTensorDescriptor_t out_desc;
        checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
        checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        d.mb, d.ic, d.ih, d.iw));
		
	
 hipdnnBatchNormMode_t bn_modeT = HIPDNN_BATCHNORM_SPATIAL;
    hipdnnTensorDescriptor_t bnScaleBiasMeanVarDescT;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDescT));
    checkHIPDNN(hipdnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDescT, out_desc, bn_modeT));

    float* bnScaleT;
    HIP_CALL(hipMalloc(&bnScaleT, 1*d.ic*1*1*sizeof(float)));
    hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,bnScaleT ,1.f);

    float* bnBiasT;
    HIP_CALL(hipMalloc(&bnBiasT, 1*d.ic*1*1*sizeof(float)));
    hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,bnBiasT ,0.f);

    float* resultRunningMean; 

    HIP_CALL(hipMalloc(&resultRunningMean, 1*d.ic*1*1*sizeof(float)));
    hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,resultRunningMean ,1.f);

    float* resultRunningVariance;
    HIP_CALL(hipMalloc(&resultRunningVariance, 1*d.ic*1*1*sizeof(float)));
    hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 , resultRunningVariance, 1.f);

    float alphaNT = 1.f;
    float betaNT = 0.f;
    double epsilonT = 2.f; 
    double exponentialAverageFactor = 0.5; 
    double* resultSaveMean = 0; 
    double* resultSaveInvVariance = 0 ;
   
    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

       timer.restart();
       
       checkHIPDNN( hipdnnBatchNormalizationForwardTraining( hipdnn, bn_modeT,
        &alphaNT, &betaNT, in_desc, src, out_desc, dst,
        bnScaleBiasMeanVarDescT, bnScaleT, bnBiasT, exponentialAverageFactor,
        resultRunningMean, resultRunningVariance, epsilonT,
        resultSaveMean, resultSaveInvVariance));

       hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
	}
	
	*avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
	
  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasMeanVarDescT);
  hipdnnDestroy(hipdnn);
 
}

#endif //TEST_BATCH_NORM_FWD_TRAINING_HPP
