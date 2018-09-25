#ifndef TEST_POOLING_COMMON_H
#define TEST_POOLING_COMMON_H

#include "hipDNN.h"
#include "hipDNN_test_common.h"

struct pool_fwd {
  int mb, c;      // Minibatch and channels
  int ih, iw;     // input dimensions
  int oh, ow;     // output dimensions
  int kh, kw;     // kernel dimensions
  int padt, padl; // padding dimensions
  int strh, strw; // stride dimensions
  pool_fwd(int mb, int c, int ih, int iw, int oh, int ow, int kh,
                     int kw, int padt, int padl, int strh, int strw)
      : mb(mb), c(c), ih(ih), iw(iw), oh(oh), ow(ow), kh(kh), kw(kw),
        padt(padt), padl(padl), strh(strh), strw(strw) {}
};

struct pool_bwd {
  size_t in, ichannel, iheight, iwidth;
  size_t wheight, wwidth;
  size_t vpadding, hpadding;
  size_t vstride, hstride;
  int on, ochannel, oheight, owidth;

  pool_bwd(size_t in, size_t ichannel, size_t iheight, size_t iwidth,
                 size_t wheight, size_t wwidth, size_t vpadding,
                 size_t hpadding, size_t vstride, size_t hstride)
      : in(in), ichannel(ichannel), iheight(iheight), iwidth(iwidth),
        wheight(wheight), wwidth(wwidth), vpadding(vpadding),
        hpadding(hpadding), vstride(vstride), hstride(hstride) {}

  pool_bwd(size_t in, size_t ichannel, size_t iheight, size_t iwidth,
                 size_t wheight, size_t wwidth, size_t vpadding,
                 size_t hpadding, size_t vstride, size_t hstride, size_t on,
                 size_t ochannel, size_t oheight, size_t owidth)
      : in(in), ichannel(ichannel), iheight(iheight), iwidth(iwidth),
        wheight(wheight), wwidth(wwidth), vpadding(vpadding),
        hpadding(hpadding), vstride(vstride), hstride(hstride), on(on),
        ochannel(ochannel), oheight(oheight), owidth(owidth) {}
};

template <typename dataType>
void hipdnn_maxpool_fwd(pool_fwd &c, dataType *src,
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


template <typename dataType>
void hipdnn_pooling_backward(pool_bwd &test_case, dataType *src,
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
      &test_case.oheight, &test_case.owidth)) hipdnnTensorDescriptor_t out_desc;
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
      hipdnnPoolingBackward(hipdnn, pool_desc, &alpha, out_desc, dst, out_desc, dst,
                        in_desc, src, &beta, in_desc, grad);
       hipDeviceSynchronize();
       std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
       time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyPoolingDescriptor(pool_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

#endif // TEST_POOLING_COMMON_H
