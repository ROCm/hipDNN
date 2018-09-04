#ifndef TEST_CONVOLUTION_FORWARD_COMMON_HPP
#define TEST_CONVOLUTION_FORWARD_COMMON_HPP

#include "hipDNN.h"
#include "hipDNN_test_common.h"
#include "gtest/gtest.h"

struct test_convolution_sizes_t {
    test_convolution_sizes_t(int mb, int ng, int ic, int ih, int iw, int oc,
                             int oh, int ow, int kh, int kw, int padh, int padw,
                             int strh, int strw, int dilh = 0, int dilw = 0)
        : mb(mb), ng(ng), ic(ic), ih(ih), iw(iw), oc(oc), oh(oh), ow(ow),
          kh(kh), kw(kw), padh(padh), padw(padw), strh(strh), strw(strw),
          dilh(dilh), dilw(dilw) {}
    int mb;         // mini batches
    int ng;         // number of groups
    int ic, ih, iw; // Input channels, height and width
    int oc, oh, ow; // Output channels, height and width
    int kh, kw;     // kernel height and width
    int padh, padw; // padding along height and width
    int strh, strw; // stride along height and width
    int dilh, dilw; // dilation along height and width
};

// Note we are currently only dealing with 2D convolution
Desc calculateConv2DOutputDesc(Desc inputDesc, Desc filterDesc, int pad[2],
                               int stride[2]) {
    assert(inputDesc.C == filterDesc.C);
    int outputHeight =
        ((inputDesc.H - filterDesc.H + 2 * pad[0]) / stride[0]) + 1;
    int outputWidth =
        ((inputDesc.W - filterDesc.W + 2 * pad[1]) / stride[1]) + 1;
    Desc outputDesc(inputDesc.N, filterDesc.N, outputHeight, outputWidth);
    return outputDesc;
}

// CPU reference code
// Input buffers are all host pointers
template <typename dataType>
void compute_cpuref_conv_fwd(test_convolution_sizes_t &c, dataType *src,
                             dataType *weights, dataType *bias, dataType *dst) {

    for (int n = 0; n < c.mb; n++) {
        for (int oc = 0; oc < c.oc; oc++) {
            for (int oh = 0; oh < c.oh; oh++) {
                for (int ow = 0; ow < c.ow; ow++) {
                    dataType a = 0; // For accumulation
                    for (int ic = 0; ic < c.ic; ic++) {
                        for (int kh = 0; kh < c.kh; kh++) {
                            for (int kw = 0; kw < c.kw; kw++) {
                                int iw = ow * c.strw - c.padw + kw * (c.dilw);
                                int ih = oh * c.strh - c.padh + kh * (c.dilh);
                                if (iw < 0 || iw >= c.iw)
                                    continue;
                                if (ih < 0 || ih >= c.ih)
                                    continue;
                                size_t iidx = n * c.ic * c.ih * c.iw +
                                              ic * c.ih * c.iw + ih * c.iw + iw;
                                size_t widx = oc * c.ic * c.kh * c.kw +
                                              ic * c.kh * c.kw + kh * c.kw + kw;
                                a += (dataType)src[iidx] * weights[widx];
                            }
                        }
                    }

                    float a_fp = (float)a;

                    a_fp += (float)(bias ? bias[oc] : 0);

                    size_t oidx = n * c.oc * c.oh * c.ow + oc * c.oh * c.ow +
                                  oh * c.ow + ow;
                    dst[oidx] = (dataType)a_fp;
                }
            }
        }
    }
}

// GPU Reference code
template <typename dataType>
void compute_hipdnn_conv_fwd(test_convolution_sizes_t &c, dataType *src,
                             dataType *weights, dataType *bias, dataType *dst) {

    hipdnnHandle_t hipdnn;
    checkHIPDNN(hipdnnCreate(&hipdnn));

    hipdnnTensorDescriptor_t in_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(in_desc, HIPDNN_TENSOR_NCHW,
                                            HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih,
                                            c.iw));

    /* float *src;
    hipMalloc(
         &src, in_n * in_c * in_h * in_w * sizeof(float));*/

    hipdnnFilterDescriptor_t filt_desc;
    checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
    int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
    checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                            HIPDNN_TENSOR_NCHW, 4, filterDimA));

    /* float *weights;
     hipMalloc(
         &weights, filt_k * filt_c * filt_h * filt_w * sizeof(float));*/

    hipdnnConvolutionDescriptor_t conv_desc;
    checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
    checkHIPDNN(hipdnnSetConvolution2dDescriptor(
        conv_desc, c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
        HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

    checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc, &c.mb, &c.oc, &c.oh, &c.ow));

    hipdnnTensorDescriptor_t out_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(out_desc, HIPDNN_TENSOR_NCHW,
                                            HIPDNN_DATA_FLOAT, c.mb, c.oc, c.oh,
                                            c.ow));

    /* float *dst;
    hipMalloc(
          &dst, out_n * out_c * out_h * out_w * sizeof(float));*/

    hipdnnConvolutionFwdAlgo_t algo;
    int MaxAlgoCount = 1;
    size_t ws_size{0};
    float *ws_data{nullptr};
    int calgo;
    hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];

    hipdnnFindConvolutionForwardAlgorithmEx(
        hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst,
        MaxAlgoCount, &calgo, algoPerf, ws_data, ws_size);
    algo = (hipdnnConvolutionFwdAlgo_t)algoPerf[0].algo;

    checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(
        hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

    hipMalloc(&ws_data, ws_size);

    // perform
    float alpha = 1.f;
    float beta = 0.f;

    checkHIPDNN(hipdnnConvolutionForward(
        hipdnn, &alpha, in_desc, src, filt_desc, weights, conv_desc, algo,
        ws_data, ws_size, &beta, out_desc, dst));

    // finalizing
    hipFree(ws_data);
    hipdnnDestroyTensorDescriptor(out_desc);
    hipdnnDestroyConvolutionDescriptor(conv_desc);
    hipdnnDestroyFilterDescriptor(filt_desc);
    hipdnnDestroyTensorDescriptor(in_desc);
    hipdnnDestroy(hipdnn);
}

#endif // TEST_CONVOLUTION_FORWARD_COMMON_HPP
