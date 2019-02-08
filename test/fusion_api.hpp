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
          HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

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
  float alphaA = 1.f;
  float betaA = 0.f;

  hipdnnConvolutionFwdPreference_t preference = HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST;
  checkHIPDNN(hipdnnGetConvolutionForwardAlgorithm( hipdnn,
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
  float betaB = 1.f;

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

  checkHIPDNN(hipdnnCreateOpConvForward( fusePlanDesc, &convOp, conv_desc, filt_desc));
  checkHIPDNN(hipdnnCreateOpBiasForward(fusePlanDesc, &biasOp, bias_desc));
  checkHIPDNN(hipdnnCreateOpActivationForward(fusePlanDesc,  &activOp, act_mode));
  checkHIPDNN(hipdnnCompileFusionPlan( hipdnn, fusePlanDesc));

  hipdnnOperatorArgs_t args;
  checkHIPDNN(hipdnnCreateOperatorArgs( &args));
  checkHIPDNN(hipdnnSetOpArgsConvForward(args, convOp, &alphaA, &betaA, weights));
  checkHIPDNN(hipdnnSetOpArgsBiasForward(args, biasOp, &alphaB, &betaB, bias_data));
  checkHIPDNN(hipdnnSetOpArgsActivForward(args, activOp, &alphaA, &betaA,
              reluCeilingOrAlpha, activBeta, activExp));

  high_resolution_timer_t timer;
  std::vector<double> time_vector(1, 0);

  for (int i = 0; i < 1; i++) {

      timer.restart();

      checkHIPDNN(hipdnnExecuteFusionPlan( hipdnn, fusePlanDesc, in_desc, src,
              out_desc, dst, args));

      hipDeviceSynchronize();

      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1000;

    }

  *avg_time = (float)std::accumulate(time_vector.begin(),
                            time_vector.end(), 0) / (benchmark_iterations);

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

template <typename dataType>
void compute_hipdnn_fusion_api_NA(convulution_Size &c, dataType *src,
                                  dataType *dst, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(in_desc, HIPDNN_TENSOR_NCHW,
               HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));

  hipdnnFusionPlanDescriptor_t fusePlanDesc;
  hipdnnFusionDirection_t fuseDirection = HIPDNN_VERTICAL_FUSION;
  hipdnnCreateFusionPlan( &fusePlanDesc, fuseDirection, in_desc);

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(out_desc, HIPDNN_TENSOR_NCHW,
               HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));

  // perform
  float alpha = 1.f;
  float beta = 0.f;

  hipdnnBatchNormMode_t BN_mode= HIPDNN_BATCHNORM_SPATIAL;

  double epsilon= 0.5e-10;
  hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc,
               HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, 1, c.oc, 1, 1));

  float * resultRunningMean;
  hipMalloc(&resultRunningMean, 1 * c.oc  *sizeof(float));

  float * resultRunningVariance;
  hipMalloc(&resultRunningVariance, 1 * c.oc * sizeof(float));

  float * bnScale;
  hipMalloc(&bnScale, 1 * c.oc *  sizeof(float));

  float * bnBias;
  hipMalloc(&bnBias, 1 * c.oc *  sizeof(float));

  hipLaunchKernelGGL(dev_const, 1 * c.oc, 1 , 0, 0 ,bnScale ,0.f);
  hipLaunchKernelGGL(dev_const, 1* c.oc, 1, 0, 0 ,bnBias, 1.f);
  hipLaunchKernelGGL(dev_const, 1*c.ic, 1*1, 0, 0 ,resultRunningMean ,0.f);
  hipLaunchKernelGGL(dev_const, 1*c.ic, 1*1, 0, 0 , resultRunningVariance,2.f);

  //Fusion

  hipdnnFusionOpDescriptor_t activOp;
  hipdnnFusionOpDescriptor_t bnOp;

  hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_RELU;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
               bnScaleBiasMeanVarDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
               1, c.oc, 1, 1));


  checkHIPDNN(hipdnnCreateOpBatchNormInference(fusePlanDesc, &bnOp, BN_mode,
                                               bnScaleBiasMeanVarDesc));

  checkHIPDNN(hipdnnCreateOpActivationForward(fusePlanDesc,  &activOp, mode));

  auto status = hipdnnCompileFusionPlan( hipdnn, fusePlanDesc);

  hipdnnOperatorArgs_t args;
  hipdnnCreateOperatorArgs( &args);

  hipdnnSetOpArgsBatchNormInference(args, bnOp, &alpha, &beta, &bnScale,
                                    &bnBias, &resultRunningMean,
                                    &resultRunningVariance, epsilon);

  double reluCeilingOrAlpha=1;
  double activBeta=1;
  double activExp=1;
  hipdnnSetOpArgsActivForward(args, activOp, &alpha, &beta,reluCeilingOrAlpha, activBeta ,activExp);

  high_resolution_timer_t timer;
  std::vector<double> time_vector(1, 0);

  for (int i = 0; i < 1; i++) {

      timer.restart();

      hipdnnExecuteFusionPlan( hipdnn, fusePlanDesc, in_desc, src, out_desc,
                               dst, args);
      hipDeviceSynchronize();

      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin(),
                                     time_vector.end(), 0)
                                     / (benchmark_iterations );

  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyOperatorArgs(args);
  hipdnnDestroyFusionPlan(fusePlanDesc);
  hipdnnDestroy(hipdnn);
}

#endif //TEST_FUSION_API_HPP
