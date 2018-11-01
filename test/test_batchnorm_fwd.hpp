#ifndef TEST_BATCHNORM_FWD_HPP
#define TEST_BATCHNORM_FWD_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"
//#define CUDNN_BN_MIN_EPSILON 1e-5
__global__ void dev_const(hipLaunchParm lp, float *px, float k) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x; 
  px[tid] = k;
}

template <typename dataType>
void compute_hipdnn_batchnorm_fwd(convulution_Size &d, dataType *src, dataType *weights,
                                dataType *dst, float *avg_time) {

	hipdnnHandle_t hipdnn;
	checkHIPDNN(hipdnnCreate(&hipdnn));
	
	hipdnnTensorDescriptor_t in_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        d.mb, d.ic, d.ih, d.iw));
		
	hipdnnFilterDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
  int filterDimA[] = {d.oc, d.ic, d.kh, d.kw};
  checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                          HIPDNN_TENSOR_NCHW, 4, filterDimA));
		
	hipdnnConvolutionDescriptor_t conv_desc;
    checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
    checkHIPDNN(hipdnnSetConvolution2dDescriptor(
        conv_desc,
        d.padh, d.padw, d.strh, d.strw, d.dilh, d.dilw,
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));
		
	checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
         &d.mb, &d.oc, &d.oh, &d.ow));
		
  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        d.mb, d.oc, d.oh, d.ow));
		
	hipdnnConvolutionFwdAlgo_t algo; 
	
    int MaxAlgoCount =1;
    size_t ws_size;
    float *ws_data;
    int calgo;
    hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];        

  hipdnnFindConvolutionForwardAlgorithmEx(hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst,
          MaxAlgoCount , &calgo, algoPerf, ws_data, ws_size);
  algo = (hipdnnConvolutionFwdAlgo_t)algoPerf[0].algo;

  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

 hipMalloc(&ws_data, ws_size);
  
  float alpha = 1.f;
  float beta = 0.f;

 checkHIPDNN(hipdnnConvolutionForward(
      hipdnn,
      &alpha, in_desc, src, filt_desc, weights,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, dst));

   hipdnnBatchNormMode_t mode= HIPDNN_BATCHNORM_SPATIAL;   //HIPDNN_BATCHNORM_PER_ACTIVATION; // HIPDNN_BATCHNORM_SPATIAL_PERSISTENT;//HIPDNN_BATCHNORM_PER_ACTIVATION;
   double exponentialAverageFactor = 1;
   double epsilon = 0.00005;
   hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        bnScaleBiasMeanVarDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        1, d.oc, 1, 1));

  float * resultRunningMean;
  hipMalloc(
        &resultRunningMean, 1 * d.oc  *sizeof(float));

  float * resultRunningVariance;
  hipMalloc(
        &resultRunningVariance, 1 * d.oc * sizeof(float));
		
  float *resultSaveMean;
  hipMalloc(
        &resultSaveMean, 1 * d.oc *  sizeof(float));
		
  float *resultSaveInvVariance;
  hipMalloc(
        &resultSaveInvVariance, 1 * d.oc * sizeof(float));
				
  float * bnScale;
  hipMalloc(
        &bnScale, 1 * d.oc *  sizeof(float));
		
  float * bnBias;
  hipMalloc(
        &bnBias, 1 * d.oc *  sizeof(float));
		
	hipLaunchKernel(dev_const, 1 * d.oc, 1 , 0, 0 ,bnScale ,0.f); //0
	hipLaunchKernel(dev_const, 1* d.oc, 1, 0, 0 ,bnBias, 1.f);  //1
		 
   // std::cout<<"\n!!!!!!!!!!!!!!!!!epsilon before function:\t"<<epsilon;
    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

       timer.restart();
       
      hipdnnBatchNormalizationForwardTraining( hipdnn,
                   mode,
      &alpha,
      &beta,
      in_desc,
      src,
      out_desc,
      dst,
      bnScaleBiasMeanVarDesc,
      bnScale,
      bnBias,
      exponentialAverageFactor,
      resultRunningMean,
      resultRunningVariance,
      epsilon,
      resultSaveMean,
      resultSaveInvVariance);

       hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
	}
	
	*avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
	// std::cout<<"\n!!!!!!!!!!!!!!!!!epsilon:\t"<<epsilon<<"\t resultSaveMean: \t"<<resultSaveMean;
  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc);
  hipdnnDestroy(hipdnn);
 
}

#endif //TEST_BATCHNORM_FWD_HPP
