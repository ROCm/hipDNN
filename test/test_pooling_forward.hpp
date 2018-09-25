#ifndef TEST_POOLING_FWD_COMMON_H
#define TEST_POOLING_FWD_COMMON_H

#include "hipDNN.h"
#include "hipDNN_test_common.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

struct test_2dpool_desc_t {
  int mb, c;      // Minibatch and channels
  int ih, iw;     // input dimensions
  int oh, ow;     // output dimensions
  int kh, kw;     // kernel dimensions
  int padt, padl; // padding dimensions
  int strh, strw; // stride dimensions
  test_2dpool_desc_t(int mb, int c, int ih, int iw, int oh, int ow, int kh,
                     int kw, int padt, int padl, int strh, int strw)
      : mb(mb), c(c), ih(ih), iw(iw), oh(oh), ow(ow), kh(kh), kw(kw),
        padt(padt), padl(padl), strh(strh), strw(strw) {}
};

template <typename dataType>
void compute_hipdnn_maxpool_fwd(test_2dpool_desc_t &c, dataType *src,
                                dataType *dst, float *avg_time) {

  hipdnnHandle_t handle;
  checkHIPDNN(hipdnnCreate(&handle));
  hipdnnTensorDescriptor_t in_desc, out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.c, c.ih, c.iw));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));
  checkHIPDNN(hipdnnSetPooling2dDescriptor(pool_desc, HIPDNN_POOLING_MAX,
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
                                   &beta, out_desc, dst));
      
       hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

  checkHIPDNN(hipdnnDestroyTensorDescriptor(in_desc));
  checkHIPDNN(hipdnnDestroyTensorDescriptor(out_desc));
  checkHIPDNN(hipdnnDestroyPoolingDescriptor(pool_desc));
  checkHIPDNN(hipdnnDestroy(handle));
}

#endif // TEST_POOLING_FWD_COMMON_H
