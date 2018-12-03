#ifndef TEST_BATCH_NORM_FWD_INFERENCE_HPP
#define TEST_BATCH_NORM_FWD_INFERENCE_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_batchnorm_fwd_inference(BNorm_params_t &d, dataType *src,
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

  hipdnnBatchNormMode_t bn_mode = HIPDNN_BATCHNORM_SPATIAL;
    /* HIPDNN_BATCHNORM_SPATIAL_PERSISTENT*/
  float alphaN = 1.f;
  float betaN = 0.f;

  hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  /*checkHIPDNN(hipdnnDeriveBNTensorDescriptor(&bnScaleBiasMeanVarDesc, out_desc, bn_mode));*/ //Only for training
  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
          bnScaleBiasMeanVarDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          1, d.ic, 1, 1));

  float* bnScale;
  HIP_CALL(hipMalloc(&bnScale, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,bnScale ,1.f);

  float* bnBias;
  HIP_CALL(hipMalloc(&bnBias, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,bnBias ,0.f);

  float* estimatedMean;
  HIP_CALL(hipMalloc(&estimatedMean, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,estimatedMean ,0.f);

  float* estimatedVariance;
  HIP_CALL(hipMalloc(&estimatedVariance, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,estimatedVariance ,2.f);

  double epsilon = 2.f; // Minimum 1e-5

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        checkHIPDNN(hipdnnnBatchNormalizationForwardInference( hipdnn, bn_mode,
                    &alphaN, &betaN, out_desc, src, out_desc , dst,
                    bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
                    estimatedVariance, epsilon));

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
      }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc);
  hipdnnDestroy(hipdnn);

}

#endif //TEST_BATCH_NORM_FWD_INFERENCE_HPP