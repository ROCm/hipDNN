#ifndef TEST_LRN_BACKWARD_HPP
#define TEST_LRN_BACKWARD_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_LRN_backward(LRN_params_t &d, dataType *src, dataType *grad,
                                dataType *dst, float *avg_time) {

	hipdnnHandle_t hipdnn;
	checkHIPDNN(hipdnnCreate(&hipdnn));
	
	hipdnnTensorDescriptor_t in_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        d.mb, d.ic, d.ih, d.iw));
		
  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        d.mb, d.ic, d.ih, d.iw));
		
  hipdnnLRNDescriptor_t lrn_desc;
    checkHIPDNN(hipdnnCreateLRNDescriptor(&lrn_desc));

    hipdnnLRNMode_t lrn_mode = HIPDNN_LRN_CROSS_CHANNEL;
    unsigned lrn_n = 5 ; // cudnn default 5  (at desc creation)
    double lrn_alpha = 0.01 ; // cudnn default 1e-4
    double lrn_beta = 0.5 ; // cudnn default 0.75
    double lrn_k = 1.0 ;  // cudnn default 2.0
    
	checkHIPDNN(hipdnnSetLRNDescriptor(lrn_desc, lrn_mode, lrn_n, lrn_alpha,
      lrn_beta, lrn_k));

    float lrn_blendAlpha = 1.f;
    float lrn_blendBeta = 0.5f;
    
	checkHIPDNN(hipdnnLRNCrossChannelForward( hipdnn, lrn_desc, lrn_mode,
        &lrn_blendAlpha, in_desc, src, &lrn_blendBeta, out_desc, dst));
    
    hipdnnTensorDescriptor_t dy_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&dy_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(
          dy_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          d.mb, d.ic, d.ih, d.iw));
    float* dy; // passed as input
    HIP_CALL(hipMalloc(&dy, d.mb*d.ic*d.ih*d.iw*sizeof(float)));
    hipLaunchKernel(dev_const, d.mb*d.ic, d.ih*d.iw, 0, 0 ,dy ,1.f);
    hipdnnTensorDescriptor_t grad_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&grad_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(
          grad_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          d.mb, d.ic, d.ih, d.iw));

   // std::cout << "grad(BackProp) BatchNorm Training Backward:" << std::endl;
   // print(grad, d.mb, d.oc, d.oh, d.ow);

  high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

       timer.restart();
       checkHIPDNN(hipdnnLRNCrossChannelBackward( hipdnn, lrn_desc, lrn_mode,
        &lrn_blendAlpha, out_desc, dst, dy_desc, dy, in_desc, src,
        &lrn_blendBeta, grad_desc, grad));
       hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
	}
	
	*avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
	
  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyLRNDescriptor(lrn_desc);
  hipdnnDestroyTensorDescriptor(dy_desc);
  hipdnnDestroyTensorDescriptor(grad_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
 
}


#endif //TEST_LRN_BACKWARD_HPP
