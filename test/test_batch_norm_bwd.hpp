#ifndef TEST_BATCH_NORM_BWD_HPP
#define TEST_BATCH_NORM_BWD_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"


__global__ void dev_const(hipLaunchParm lp, float *px, float k) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x; 
  px[tid] = k;
}

__global__ void dev_iota(hipLaunchParm lp, float *px) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  px[tid] = tid + 1;
}

__global__ void dev_iota_1(hipLaunchParm lp, float *px) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  px[tid] = tid + 1;
}

template <typename dataType>
void compute_hipdnn_batchnorm_bwd(BNorm_params_t &d, dataType *src,
                                dataType *dx, dataType *resultBnScaleDiff, dataType *resultBnBiasDiff, float *avg_time, int flag) {
 
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
    
    hipdnnTensorDescriptor_t bnScaleBiasDiffDesc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasDiffDesc));
    hipdnnBatchNormMode_t bn_modeT_back;
    
    if (flag == 0 || flag == 2)
    
       bn_modeT_back = HIPDNN_BATCHNORM_SPATIAL;
        
    
    else
    
        bn_modeT_back = HIPDNN_BATCHNORM_PER_ACTIVATION;
      
   

    checkHIPDNN(hipdnnDeriveBNTensorDescriptor(bnScaleBiasDiffDesc, in_desc, bn_modeT_back));

    int out_n,out_c,out_h,out_w, nStride,cStride,hStride,wStride; 
    hipdnnDataType_t dt = HIPDNN_DATA_FLOAT;
    hipdnnGetTensor4dDescriptor(
    bnScaleBiasDiffDesc,
    &dt,
    &out_n,
    &out_c,
    &out_h,
    &out_w,
    &nStride,
    &cStride,
    &hStride,
    &wStride); 

    hipdnnTensorDescriptor_t dy_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&dy_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(
          dy_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          d.mb, d.ic, d.ih, d.iw));

    float* dy; 
    HIP_CALL(hipMalloc(&dy, d.mb*d.ic*d.ih*d.iw*sizeof(float)));
    hipLaunchKernel(dev_iota, d.iw * d.ih, d.mb * d.ic , 0, 0 ,dy);
    hipdnnTensorDescriptor_t dx_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&dx_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(
          dx_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          d.mb, d.ic, d.ih, d.iw));

    float* bnScaleT_back;
    HIP_CALL(hipMalloc(&bnScaleT_back, out_h * out_w * out_c * sizeof(float)));
    hipLaunchKernel(dev_const, out_w * out_h, out_c , 0, 0 ,bnScaleT_back ,1.f);
    
    float alphaDataDiff = 1.f;
    float betaDataDiff = 0.f;
    float alphaParamDiff = 1.f;
    float betaParamDiff = 0.f;
    double epsilonT_back = 1e-5;
    
    float *savedMean;
  hipMalloc(
        &savedMean, out_h * out_w * out_c * sizeof(float));
  hipLaunchKernel(dev_iota_1, out_w * out_h,  out_c , 0, 0 ,savedMean);  
		
  float *savedInvVariance;
  hipMalloc(
        &savedInvVariance, out_h * out_w * out_c * sizeof(float));
  hipLaunchKernel(dev_iota, out_w * out_h,  out_c , 0, 0 ,savedInvVariance);

   high_resolution_timer_t timer;
   std::vector<double> time_vector(benchmark_iterations, 0);

   for (int i = 0; i < benchmark_iterations; i++) {

   timer.restart();

   	 checkHIPDNN( hipdnnBatchNormalizationBackward( hipdnn, bn_modeT_back,
      		&alphaDataDiff, &betaDataDiff, &alphaParamDiff, &betaParamDiff,
     		 in_desc, src, dy_desc, dy, dx_desc, dx, bnScaleBiasDiffDesc,
      		bnScaleT_back, resultBnScaleDiff, resultBnBiasDiff,
      		epsilonT_back, savedMean, savedInvVariance));

         if (flag == 2 || flag == 3)
            {
              betaDataDiff = 1.f;
              betaParamDiff = 1.f;
      
              checkHIPDNN( hipdnnBatchNormalizationBackward( hipdnn, bn_modeT_back,
      		&alphaDataDiff, &betaDataDiff, &alphaParamDiff, &betaParamDiff,
     		 in_desc, src, dy_desc, dy, dx_desc, dx, bnScaleBiasDiffDesc,
      		bnScaleT_back, resultBnScaleDiff, resultBnBiasDiff,
      		epsilonT_back, savedMean, savedInvVariance));
            }

  hipDeviceSynchronize();
  std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
  time_vector[i] = (double)time_elapsed / 1000;
  }
	
   *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);


  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasDiffDesc);
  hipdnnDestroy(hipdnn);
 
}

#endif //TEST_BATCH_NORM_BWD_HPP
