#ifndef TEST_FUSION_API_HPP
#define TEST_FUSION_API_HPP

#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_fusion_api(convulution_Size &c, dataType *src,
                             dataType *weights, dataType *bias_data,
                             dataType *dst, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
          in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          c.mb, c.ic, c.ih, c.iw));

  int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
  hipdnnFilterDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
  checkHIPDNN(hipdnnSetFilterNdDescriptor(
          filt_desc, HIPDNN_DATA_FLOAT,HIPDNN_TENSOR_NCHW,
          4, filterDimA));

  hipdnnConvolutionDescriptor_t conv_desc;
  checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
  checkHIPDNN(hipdnnSetConvolution2dDescriptor(
          conv_desc, c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
          HIPDNN_CONVOLUTION, HIPDNN_DATA_FLOAT));

  checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
          conv_desc, in_desc, filt_desc,
          &c.mb, &c.oc, &c.oh, &c.ow));
  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
          out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          c.mb, c.oc, c.oh, c.ow));

  hipdnnConvolutionFwdAlgo_t algo;
  size_t ws_size;
  float *ws_data;
  float alphaC = 1.f;
  float betaC = 0.f;

  hipdnnConvolutionFwdPreference_t preference = HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST;
  CHECK_HIPDNN(hipdnnGetConvolutionForwardAlgorithm( hipdnn,
          in_desc, filt_desc, conv_desc, out_desc, preference,
          0 /*memoryLimitInBytes*/ ,&algo));
                                      // WorkspaceSize to be based on preference

  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize( hipdnn, in_desc, filt_desc,
          conv_desc, out_desc, algo, &ws_size));
  HIP_CALL(hipMalloc(&ws_data, ws_size));

  hipdnnTensorDescriptor_t bias_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&bias_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
          bias_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          c.mb, c.oc , c.oh, c.ow));

  float alphaB = 1.f;
  float betaB = 0.f;
  checkHIPDNN(hipdnnAddTensor(hipdnn, &alphaB, bias_desc, bias_data, &betaB,
                              out_desc, dst));

  float alphaA = 1.f;
  float betaA = 0.f;

  hipdnnActivationDescriptor_t activationDesc;
  hipdnnActivationMode_t act_mode = HIPDNN_ACTIVATION_RELU;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;

  double reluCeilingOrAlpha=0;
  double activBeta=0;
  double activExp=0;

  checkHIPDNN(hipdnnCreateActivationDescriptor(&activationDesc));
  checkHIPDNN(hipdnnSetActivationDescriptor(activationDesc, act_mode,
              reluNanOpt,reluCeilingOrAlpha, activBeta, activExp));

  hipdnnFusionPlanDescriptor_t fusePlanDesc;
  hipdnnFusionDirection_t fuseDirection = HIPDNN_VERTICAL_FUSION;
  checkHIPDNN(hipdnnCreateFusionPlan( &fusePlanDesc, fuseDirection, in_desc));

  hipdnnFusionOpDescriptor_t convOp;
  hipdnnFusionOpDescriptor_t biasOp;
  hipdnnFusionOpDescriptor_t activOp;
  hipdnnFusionOpDescriptor_t bnOp;

  checkHIPDNN(hipdnnCreateOpConvForward( fusePlanDesc, &convOp, conv_desc, filt_desc));
  checkHIPDNN(hipdnnCreateOpBiasForward(fusePlanDesc, &biasOp, bias_desc));
  checkHIPDNN(hipdnnCreateOpActivationForward(fusePlanDesc,  &activOp, act_mode));
  checkHIPDNN(hipdnnCompileFusionPlan( hipdnn, fusePlanDesc));

  hipdnnOperatorArgs_t args;
  checkHIPDNN(hipdnnCreateOperatorArgs( &args));
  checkHIPDNN(hipdnnSetOpArgsConvForward(args, convOp, &alphaC, &betaC, weights));
  checkHIPDNN(hipdnnSetOpArgsBiasForward(args, biasOp, &alphaB, &betaB, bias_data));
  checkHIPDNN(hipdnnSetOpArgsActivForward(args, activOp, &alphaA, &betaA,
              reluCeilingOrAlpha, activBeta, activExp));

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

      timer.restart();

      checkHIPDNN(hipdnnExecuteFusionPlan( hipdnn, fusePlanDesc, in_desc, src,
              out_desc, dst, args));
      hipDeviceSynchronize();

      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                                     time_vector.end(), 0)
                                     / (benchmark_iterations - 10);

  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyTensorDescriptor(bias_desc);
  hipdnnDestroyActivationDescriptor(activationDesc);
  hipdnnDestroyOperatorArgs(args);
  hipdnnDestroyFusionPlan(fusePlanDesc);
  hipdnnDestroy(hipdnn);

}

#endif //TEST_FUSION_API_HPP