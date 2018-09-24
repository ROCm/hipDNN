#include "hipDNN_test_common.h"

struct activation_params_t {
  int n, channels, height, width;
  activation_params_t(int n, int channels, int height, int width)
      : n(n), channels(channels), height(height), width(width) {}
};

template <typename dataType>
void compute_hipdnn_activation_backward(activation_params_t &test_case,
                                        dataType *src, dataType *grad,
                                        dataType *dst) {
  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));
  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));
  hipdnnActivationDescriptor_t activationDesc;
  //hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_SIGMOID;
  hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_RELU;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;
  double reluCeilingOrAlpha = 0;
  double activBeta = 0;
  double activExp = 0;
  checkHIPDNN(hipdnnCreateActivationDescriptor(&activationDesc));
  checkHIPDNN(hipdnnSetActivationDescriptor(activationDesc, mode, reluNanOpt,
                                            reluCeilingOrAlpha, activBeta,
                                            activExp));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));
  float alpha = 1.f;
  float beta = 0.f;

  hipdnnActivationForward(hipdnn, activationDesc, &alpha, in_desc, src, &beta,
                          out_desc, dst);

  hipdnnActivationBackward(hipdnn, activationDesc, &alpha, in_desc, src,
                           in_desc, src, out_desc, dst, &beta, out_desc, grad);

  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyActivationDescriptor(activationDesc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}
