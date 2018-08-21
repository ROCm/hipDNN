#ifndef TEST_CONVOLUTION_FORWARD_COMMON_HPP
#define TEST_CONVOLUTION_FORWARD_COMMON_HPP

#include "gtest/gtest.h"
#include "hipDNN_test_common.h"

template<typename dataType>
void compute_cpuref_conv_fwd(test_convolution_sizes_t& c, dataType* src, dataType* weights, dataType* bias, dataType* dst) {

    for (int n = 0; n < c.mb; n++) {
        for (int oc = 0; oc < c.oc; oc++) {
            for (int oh = 0; oh < c.oh; oh++) {
                for (int ow = 0; ow < c.ow; ow++) {
                    dataType a = 0; // For accumulation
                    for (int ic = 0; ic < c.ic; ic++) {
                        for (int kh = 0; kh < c.kh; kh++) {
                            for (int kw = 0; kw < c.kw; kw++) {
                                int iw = ow * c.strw
                                      - c.padw + kw * (1 + c.dilw);
                                int ih = oh * c.strh
                                      - c.padh + kh * (1 + c.dilh);
                                if (iw < 0 || iw >= c.iw) continue;
                                if (ih < 0 || ih >= c.ih) continue;
                                size_t iidx = n * c.ic * c.ih * c.iw
                                    + ic * c.ih * c.iw + ih * c.iw + iw;
                                size_t widx = oc * c.ic  * c.kh * c.kw
                                    + ic * c.kh * c.kw + kh * c.kw + kw;
                                a += (dataType)src[iidx *  weights_data[widx];
                            }
                        }
                    }

                    float a_fp = (float)a;

                    a_fp += (float)(bias ?
                        bias[c.oc + oc)] :
                        0);

                    size_t oidx = n * c.oc * c.oh * c.ow
                             + oc * c.oh * c.ow + oh * c.ow + ow;
                    dst[oidx] = (dataType)a_fp;
                }
            }
        }
    }
}

#endif //TEST_CONVOLUTION_FORWARD_COMMON_HPP
