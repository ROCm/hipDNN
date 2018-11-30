#ifndef TEST_FUSION_API__NA_HPP
#define TEST_FUSION_API__NA_HPP

#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_fusion_api_NA
(convulution_Size &c, dataType *src,
                             dataType *dst, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));
		
  hipdnnFusionPlanDescriptor_t fusePlanDesc;
  hipdnnFusionDirection_t fuseDirection = HIPDNN_VERTICAL_FUSION;
  hipdnnCreateFusionPlan( &fusePlanDesc,
                       fuseDirection,
                       in_desc);

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        c.mb, c.ic, c.ih, c.iw));

  // perform
  float alpha = 1.f;
  float beta = 0.f;

   hipdnnBatchNormMode_t BN_mode= HIPDNN_BATCHNORM_SPATIAL_PERSISTENT;//HIPDNN_BATCHNORM_PER_ACTIVATION;
   double exponentialAverageFactor = 2;
   double epsilon= 0.5e-10;
   hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc;

   checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
   checkHIPDNN(hipdnnSetTensor4dDescriptor(
        bnScaleBiasMeanVarDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        1, c.oc, 1, 1));

   float * resultRunningMean;
   hipMalloc(
        &resultRunningMean, 1 * c.oc  *sizeof(float));

   float * resultRunningVariance;
   hipMalloc(
        &resultRunningVariance, 1 * c.oc * sizeof(float));
		
   float *resultSaveMean;
   hipMalloc(
        &resultSaveMean, 1 * c.oc *  sizeof(float));
		
   float *resultSaveInvVariance;
   hipMalloc(
        &resultSaveInvVariance, 1 * c.oc * sizeof(float));
				
   float * bnScale;
   hipMalloc(
        &bnScale, 1 * c.oc *  sizeof(float));
		
   float * bnBias;
   hipMalloc(
        &bnBias, 1 * c.oc *  sizeof(float));
		
   hipLaunchKernel(dev_const, 1 * c.oc, 1 , 0, 0 ,bnScale ,0.f);
   hipLaunchKernel(dev_const, 1* c.oc, 1, 0, 0 ,bnBias, 1.f);
	  
  //Fusion

	hipdnnFusionOpDescriptor_t activOp;
	hipdnnFusionOpDescriptor_t bnOp;
	
	 hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_RELU;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        bnScaleBiasMeanVarDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        1, c.oc, 1, 1));

		
	checkHIPDNN(hipdnnCreateOpBatchNormInference(fusePlanDesc, &bnOp, BN_mode, bnScaleBiasMeanVarDesc));
	checkHIPDNN(hipdnnCreateOpActivationForward(fusePlanDesc,  &activOp, mode));

 	 auto status = hipdnnCompileFusionPlan( hipdnn, fusePlanDesc);
	
	hipdnnOperatorArgs_t args;
    hipdnnCreateOperatorArgs( &args);
	
    hipdnnSetOpArgsBatchNormInference(args, bnOp, &alpha, &beta, &bnScale, &bnBias, &resultRunningMean, &resultRunningVariance, epsilon);
    hipdnnSetOpArgsActivForward(args, activOp, &alpha,&beta,1, 1 ,1);
           
    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        hipdnnExecuteFusionPlan( hipdnn, fusePlanDesc, in_desc, src, out_desc, dst, args); 
        hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
  
  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyOperatorArgs(args);
  hipdnnDestroyFusionPlan(fusePlanDesc);
  hipdnnDestroy(hipdnn); 
}

#endif //TEST_FUSION_API__NA_HPP
