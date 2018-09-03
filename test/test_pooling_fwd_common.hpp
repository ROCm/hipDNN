#ifndef TEST_POOLING_FWD_COMMON_H
#define TEST_POOLING_FWD_COMMON_H

#include "gtest/gtest.h"
#include "hipDNN_test_common.h"
#include "hipDNN.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <vector>


enum {
    MAXPOOL,
    AVGPOOL
}PoolType;


struct test_2dpool_desc_t {
    int mb, c; // Minibatch and channels
    int ih, iw; // input dimensions
    int oh, ow; // output dimensions
    int kh, kw; // kernel dimensions
    int padt, padl; // padding dimensions
    int strh, strw; // stride dimensions
    test_2dpool_desc_t(int mb, int c, 
                       int ih, int iw, 
                       int oh, int ow,
                       int kh, int kw,
                       int padt, int padl,
                       int strh, int strw
                    ): mb(mb), c(c),
                       ih(ih), iw(iw),
                       oh(oh), ow(ow),
                       kh(kh), kw(kw),
                       padt(padt), padl(padl),
                       strh(strh), strw(strw){
                           
                       }
};

template <typename dataType> struct acc_t { typedef dataType type; };


template<typename dataType>
void compute_cpuref_maxpool_fwd(test_2dpool_desc_t pd, dataType* src, dataType* dst){
     for (int n = 0; n < pd.mb; n++) {
         for (int c = 0; c < pd.c; c++) {
             for (int oh = 0; oh < pd.oh; oh++) {
                for (int ow = 0; ow < pd.ow; ow++) {
                     size_t oidx = (size_t)n * pd.c * pd.oh * pd.ow
                             + (size_t)c * pd.oh * pd.ow
                             + (size_t)oh * pd.ow + ow;

                     typename acc_t<dataType>::type acc_ref
                             = std::numeric_limits<dataType>::lowest();

                     int out_ref_index = 0;
                     bool is_initialized = false;


                    for (int kh = 0; kh < pd.kh; ++kh)
                    for (int kw = 0; kw < pd.kw; ++kw)
                     {
                         const int ih = oh * pd.strh - pd.padt + kh;
                         const int iw = ow * pd.strw - pd.padl + kw;

                         if (ih < 0 || ih >= pd.ih) continue;
                         if (iw < 0 || iw >= pd.iw) continue;

                         size_t iidx
                                 = (size_t)n * pd.c * pd.ih * pd.iw
                                 + (size_t)c * pd.ih * pd.iw
                                 + (size_t)ih * pd.iw + iw;

                         dataType d = src[iidx];

                         if (!is_initialized) {
                             acc_ref = d;
                             out_ref_index = kh * pd.kw + kw;
                             is_initialized = true;
                         } else {
                             if (acc_ref < d) {
                                 acc_ref = d;
                                 out_ref_index = kh * pd.kw + kw;
                             }
                         }
                     }
                     const dataType out_ref = (dataType)acc_ref;
                     dst[oidx] = out_ref;
                }
             }
         }
     }
 }

template<typename dataType>
void compute_hipdnn_maxpool_fwd(test_2dpool_desc_t &c, dataType *src, dataType *dst){
    hipdnnHandle_t handle;
    checkHIPDNN(hipdnnCreate(&handle));
    hipdnnTensorDescriptor_t in_desc, out_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(in_desc, HIPDNN_TENSOR_NCHW,
                                            HIPDNN_DATA_FLOAT, c.mb, c.c, c.ih,
                                            c.iw));

    hipdnnPoolingDescriptor_t pool_desc;
    checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));
    checkHIPDNN(hipdnnSetPooling2dDescriptor(
        pool_desc, HIPDNN_POOLING_MAX,
        HIPDNN_NOT_PROPAGATE_NAN, c.kw, c.kh, c.padt, c.padl, c.strh, c.strw
    ));
    checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &c.mb, &c.c, &c.oh, &c.ow));

    checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(out_desc, HIPDNN_TENSOR_NCHW,
                                            HIPDNN_DATA_FLOAT, c.mb, c.c, c.oh,
                                            c.ow));
    printf("%d, %d, %d, %d\n", c.mb, c.c, c.oh, c.ow);
    float alpha = 1.f;
    float beta = 0.f;

    checkHIPDNN(hipdnnPoolingForward(
        handle,
        pool_desc,
        &alpha,
        in_desc,
        src,
        &beta,
        out_desc,
        dst
    ));

    checkHIPDNN(hipdnnDestroyTensorDescriptor(in_desc));
    checkHIPDNN(hipdnnDestroyTensorDescriptor(out_desc));
    checkHIPDNN(hipdnnDestroyPoolingDescriptor(pool_desc));
    checkHIPDNN(hipdnnDestroy(handle));
}



#endif // TEST_POOLING_FWD_COMMON_H


