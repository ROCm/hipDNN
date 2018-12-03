#ifndef POOLING_BACKWARD
#define POOLING_BACKWARD
#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_pooling_backward(pool_bwd &test_case, dataType *src,
                               dataType *grad, dataType *dst, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.in,
      test_case.ichannel, test_case.iheight, test_case.oheight));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));

  hipdnnPoolingMode_t poolmode = HIPDNN_POOLING_MAX;
  hipdnnNanPropagation_t maxpoolingNanOpt = HIPDNN_NOT_PROPAGATE_NAN;

  checkHIPDNN(hipdnnSetPooling2dDescriptor(
      pool_desc, poolmode, maxpoolingNanOpt, test_case.wheight,
      test_case.wwidth, test_case.vpadding, test_case.hpadding,
      test_case.vstride, test_case.hstride));

  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(
      pool_desc, in_desc, &test_case.on, &test_case.ochannel,
      &test_case.oheight, &test_case.owidth));
  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.on,
      test_case.ochannel, test_case.oheight, test_case.owidth));

  float alpha = 1.f;
  float beta = 0.f;

  hipdnnPoolingForward(hipdnn, pool_desc, &alpha, in_desc, src, &beta, out_desc,
                       dst);

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        hipdnnPoolingBackward(hipdnn, pool_desc, &alpha, out_desc, dst, out_desc,
                        dst, in_desc, src, &beta, in_desc, grad);

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                    0) / (benchmark_iterations - 10);


  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyPoolingDescriptor(pool_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);

 }

#endif