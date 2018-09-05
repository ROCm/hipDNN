#include "hipDNN_test_common.h"

struct test_pooling_t{    
    size_t in, ichannel, iheight, iwidth;
    size_t wheight, wwidth;
    size_t vpadding, hpadding;
    size_t vstride, hstride;
    int on, ochannel, oheight, owidth;

    test_pooling_t(size_t in, size_t  ichannel, size_t iheight, size_t iwidth,
                   size_t wheight, size_t wwidth,
                   size_t vpadding, size_t hpadding,
                   size_t vstride, size_t hstride): in(in), ichannel(ichannel), iheight(iheight), iwidth(iwidth), wheight(wheight), wwidth(wwidth),
                   vpadding(vpadding), hpadding(hpadding), vstride(vstride),
                   hstride(hstride){}

    test_pooling_t(size_t in, size_t  ichannel, size_t iheight, size_t iwidth,
                   size_t wheight, size_t wwidth,
                   size_t vpadding, size_t hpadding,
                   size_t vstride, size_t hstride,
                   size_t on, size_t ochannel, size_t oheight, size_t owidth): in(in), ichannel(ichannel), iheight(iheight), iwidth(iwidth), wheight(wheight), wwidth(wwidth),
                   vpadding(vpadding), hpadding(hpadding), vstride(vstride),
                   hstride(hstride), on(on), ochannel(ochannel), oheight(oheight), owidth(owidth){}
};

template <typename dataType>
void compute_hipdnn_pooling_backward(test_pooling_t &test_case, dataType *src, dataType *grad, dataType *dst){
  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));
  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        test_case.in, test_case.ichannel, test_case.iheight, test_case.oheight));
  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));

  hipdnnPoolingMode_t poolmode = HIPDNN_POOLING_MAX;
  hipdnnNanPropagation_t maxpoolingNanOpt= HIPDNN_NOT_PROPAGATE_NAN ;
 
  checkHIPDNN(hipdnnSetPooling2dDescriptor(
        pool_desc, poolmode,
        maxpoolingNanOpt, test_case.wheight,
        test_case.wwidth, test_case.vpadding,  test_case.hpadding, test_case.vstride, test_case.hstride));
      
  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(pool_desc,
                                                in_desc,
                                                &test_case.on,
                                                &test_case.ochannel,
                                                &test_case.oheight,
                                                &test_case.owidth))
  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        test_case.on, test_case.ochannel, test_case.oheight, test_case.owidth));
  float alpha = 1.f;
  float beta = 0.f;

    hipdnnPoolingForward( hipdnn, pool_desc, &alpha,  in_desc, src, &beta, out_desc, dst);

    hipdnnPoolingBackward( hipdnn, pool_desc, &alpha, out_desc, dst, out_desc, dst, in_desc, 
                       src, &beta, in_desc, grad);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyPoolingDescriptor(pool_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);    
}