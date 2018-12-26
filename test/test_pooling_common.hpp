#ifndef TEST_POOLING_COMMON_HPP
#define TEST_POOLING_COMMON_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void hipdnn_pooling_forward(test_pooling_descriptor &c, dataType *src,
                                dataType *dst, hipdnnPoolingMode_t mode,
                                bool do_backward,
                                float *avg_time) {

  hipdnnHandle_t handle;
  checkHIPDNN(hipdnnCreate(&handle));

  hipdnnTensorDescriptor_t in_desc, out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.c, c.ih, c.iw));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));
  checkHIPDNN(hipdnnSetPooling2dDescriptor(pool_desc, mode,
                                           HIPDNN_NOT_PROPAGATE_NAN, c.kw, c.kh,
                                           c.padt, c.padl, c.strh, c.strw));

  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &c.mb,
                                                 &c.c, &c.oh, &c.ow));

  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.c, c.oh, c.ow));

  float alpha = 1.f;
  float beta = 0.f;

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        checkHIPDNN(hipdnnPoolingForward(handle, pool_desc, &alpha, in_desc, src,
                                   &beta, out_desc, dst, do_backward))

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                    0) / (benchmark_iterations - 10);

  checkHIPDNN(hipdnnDestroyTensorDescriptor(in_desc));
  checkHIPDNN(hipdnnDestroyTensorDescriptor(out_desc));
  checkHIPDNN(hipdnnDestroyPoolingDescriptor(pool_desc));
  checkHIPDNN(hipdnnDestroy(handle));

}

template <typename dataType>
void hipdnn_pooling_backward(pool_bwd &test_case, dataType *src,
                               dataType *grad, dataType *dst,
                               hipdnnPoolingMode_t mode,
                               float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));

  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.in,
      test_case.ichannel, test_case.iheight, test_case.oheight));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));

  hipdnnNanPropagation_t maxpoolingNanOpt = HIPDNN_NOT_PROPAGATE_NAN;

  checkHIPDNN(hipdnnSetPooling2dDescriptor(
      pool_desc, mode, maxpoolingNanOpt, test_case.wheight,
      test_case.wwidth, test_case.vpadding, test_case.hpadding,
      test_case.vstride, test_case.hstride));

  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(
      pool_desc, in_desc, &test_case.on, &test_case.ochannel,
      &test_case.oheight, &test_case.owidth)) hipdnnTensorDescriptor_t out_desc;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));

  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.on,
      test_case.ochannel, test_case.oheight, test_case.owidth));

  float alpha = 1.f;
  float beta = 0.f;

  hipdnnPoolingForward(hipdnn, pool_desc, &alpha, in_desc, src, &beta, out_desc,
                       dst, true);

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

#endif // TEST_POOLING_COMMON_H
