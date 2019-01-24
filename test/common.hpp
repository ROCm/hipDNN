#ifndef COMMON_HPP
#define COMMON_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"
#include "hip/hip_runtime_api.h"
#include "hip/hip_fp16.h"

extern float alpha;
extern float beta;
extern float alpha_beta[4];
extern hipdnnActivationMode_t act_mode;
extern hipdnnPoolingMode_t pool_mode;

inline Desc calculate_Dims(Desc inputDesc, Desc filterDesc, int pad[2],
                               int stride[2], int dilution[2]) {
  assert(inputDesc.C == filterDesc.C);
  int outputHeight = ((inputDesc.H - filterDesc.H + 2 * pad[0] -
                       (filterDesc.H - 1)*(dilution[0] -1)) / stride[0]) + 1;
  int outputWidth = ((inputDesc.W - filterDesc.W + 2 * pad[1] -
                     (filterDesc.H -1)*(dilution[1] -1)) / stride[1]) + 1;
  Desc outputDesc(inputDesc.N, filterDesc.N, outputHeight, outputWidth);
  return outputDesc;
}

inline Desc calculate_pool_Dims(Desc inputDesc, int spatial_ext[2], int pad[2],
                                int stride[2]) {
  int outputHeight = ((inputDesc.H - spatial_ext[0] + 2 * pad[0]) / stride[0]) + 1;
  int outputWidth = ((inputDesc.W - spatial_ext[1] + 2 * pad[1]) / stride[1]) + 1;
  Desc outputDesc(inputDesc.N, inputDesc.C, outputHeight, outputWidth);
  return outputDesc;
}

struct convulution_Size {
  convulution_Size(int mb, int ng, int ic, int ih, int iw, int oc,
                           int oh, int ow, int kh, int kw, int padh, int padw,
                           int strh, int strw, int dilh = 0, int dilw = 0)
      : mb(mb), ng(ng), ic(ic), ih(ih), iw(iw), oc(oc), oh(oh), ow(ow), kh(kh),
        kw(kw), padh(padh), padw(padw), strh(strh), strw(strw), dilh(dilh),
        dilw(dilw) {}
  int mb;         // mini batches
  int ng;         // number of groups
  int ic, ih, iw; // Input channels, height and width
  int oc, oh, ow; // Output channels, height and width
  int kh, kw;     // kernel height and width
  int padh, padw; // padding along height and width
  int strh, strw; // stride along height and width
  int dilh, dilw; // dilation along height and width
};

struct test_pooling_descriptor {
  int mb, c;      // Minibatch and channels
  int ih, iw;     // input dimensions
  int oh, ow;     // output dimensions
  int kh, kw;     // kernel dimensions
  int padt, padl; // padding dimensions
  int strh, strw; // stride dimensions
  test_pooling_descriptor(int mb, int c, int ih, int iw, int oh, int ow, int kh,
                     int kw, int padt, int padl, int strh, int strw)
      : mb(mb), c(c), ih(ih), iw(iw), oh(oh), ow(ow), kh(kh), kw(kw),
        padt(padt), padl(padl), strh(strh), strw(strw) {}
};

struct activation_params_t {
  int n, channels, height, width;
  activation_params_t(int n, int channels, int height, int width)
      : n(n), channels(channels), height(height), width(width) {}
};

struct BNorm_params_t {
  int mb, ic, ih, iw;
  BNorm_params_t(int mb, int ic, int ih, int iw)
      : mb(mb), ic(ic), ih(ih), iw(iw) {}
};

struct LRN_params_t {
  int mb, ic, ih, iw;
  LRN_params_t(int mb, int ic, int ih, int iw)
      : mb(mb), ic(ic), ih(ih), iw(iw) {}
};

#endif //COMMON_HPP