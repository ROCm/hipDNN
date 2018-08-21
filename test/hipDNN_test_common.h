#ifndef HIPDNN_TEST_COMMON_HPP
#define HIPDNN_TEST_COMMON_HPP

#include "hipDNN.h"
#include <cstdlib>

struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb; // mini batches
    int ng; // number of groups
    int ic, ih, iw;  // Input channels, height and width
    int oc, oh, ow;  // Output channels, height and width
    int kh, kw;  // kernel height and width
    int padh, padw; // padding along height and width
    int strh, strw; // stride along height and width
    int dilh, dilw; // dilation along height and width
};


#define checkHIPDNN(expression)                               \
  {                                                          \
    hipdnnStatus_t status = (expression);                     \
    if (status != HIPDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << hipdnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define HIP_CALL(f) { \
  hipError_t err = (f); \
  if (err != hipSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}


#endif
