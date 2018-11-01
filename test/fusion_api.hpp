#ifndef TEST_FUSION_API_HPP
#define TEST_FUSION_API_HPP

#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_fusion_api(convulution_Size &c, dataType *src,
                             dataType *weights, dataType *bias_data, dataType *dst, float *avg_time) {

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
	
  hipdnnTensorDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&filt_desc));
  int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
  checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                          HIPDNN_TENSOR_NCHW, 4, filterDimA));

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
	  
  hipdnnTensorDescriptor_t bias_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&bias_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
       bias_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
       c.mb, c.oc, c.oh, c.ow));
  
//    hipdnnConvolutionFwdAlgo_t algo; 
    int MaxAlgoCount =5;
    size_t ws_size;
    float *ws_data;
    int calgo;
    
  // perform
  float alpha = 1.f;
  float beta = 0.f;
  beta = 0.f; 

  hipdnnActivationDescriptor_t activationDesc;
    hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_RELU;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;
  double coef = 0;
  double reluCeilingOrAlpha=0;
  double activBeta=0;
  double activExp=0;
  checkHIPDNN(hipdnnCreateActivationDescriptor(&activationDesc));
  checkHIPDNN(hipdnnSetActivationDescriptor(activationDesc, mode,
            reluNanOpt,reluCeilingOrAlpha, activBeta, activExp));
 
 
  //Fusion
    algo =  HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;//HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;


  
	hipdnnFusionOpDescriptor_t convOp;		
	hipdnnFusionOpDescriptor_t biasOp;
	hipdnnFusionOpDescriptor_t activOp;
	mode = HIPDNN_ACTIVATION_RELU;
	checkHIPDNN(hipdnnCreateOpConvForward( fusePlanDesc, &convOp, conv_desc, filt_desc));
	checkHIPDNN(hipdnnCreateOpBiasForward(fusePlanDesc, &biasOp, bias_desc));
	checkHIPDNN(hipdnnCreateOpActivationForward(fusePlanDesc,  &activOp, mode));
	
	/*hipdnnFusionPlanConvolutionGetAlgo( fusePlanDesc, 4, &calgo, &algo);*/
  size_t workSpaceSize;
	hipdnnFusionPlanGetWorkSpaceSize(hipdnn, fusePlanDesc, &workSpaceSize, algo);
	std::cout<<"\nALGO:"<<algo<<"\t"<<workSpaceSize;
	
 	 auto status = hipdnnCompileFusionPlan( hipdnn, fusePlanDesc);
	hipdnnOperatorArgs_t args;
    hipdnnCreateOperatorArgs( &args);
    hipdnnSetOpArgsConvForward(args, convOp, &alpha, &beta, weights);
	hipdnnSetOpArgsBiasForward(args, biasOp,&alpha, &beta, bias_data);
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
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyTensorDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyOperatorArgs(args);
  hipdnnDestroyFusionPlan(fusePlanDesc);
  hipdnnDestroy(hipdnn);
}

#endif //TEST_FUSION_API_HPP
