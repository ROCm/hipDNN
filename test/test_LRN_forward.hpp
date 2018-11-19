#ifndef TEST_LRN_FWD_COMMON_H

#define TEST_LRN_FWD_COMMON_H

#include "hipdnn.h"

#include "hipdnn_test_common.h"

#include "gtest/gtest.h"

#include <algorithm>

#include <functional>

#include <iostream>

#include <iterator>

#include <random>

#include <vector>

#include "common.hpp"



template <typename dataType>

void compute_hipdnn_LRN_fwd(LRN_params_t &d, dataType *src, dataType *dst, float *avg_time) {



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

  hipdnnCreateLRNDescriptor(

           &lrn_desc);

  

  hipdnnLRNMode_t lrn_mode = HIPDNN_LRN_CROSS_CHANNEL;

    unsigned lrn_n = 5 ; // cudnn default 5  (at desc creation)

    double lrn_alpha = 0.01 ; // cudnn default 1e-4

    double lrn_beta = 0.5 ; // cudnn default 0.75

    double lrn_k = 1.0 ;  // cudnn default 2.0

  checkHIPDNN(hipdnnSetLRNDescriptor(lrn_desc, lrn_mode, lrn_n, lrn_alpha,

      lrn_beta, lrn_k));



    float lrn_blendAlpha = 1.f;

    float lrn_blendBeta = 0.5f;

  

  high_resolution_timer_t timer;

    std::vector<double> time_vector(benchmark_iterations, 0);


      for (int i = 0; i < benchmark_iterations; i++) {

           timer.restart();

           checkHIPDNN(hipdnnLRNCrossChannelForward( hipdnn, lrn_desc, lrn_mode,

                       &lrn_blendAlpha, in_desc, src, &lrn_blendBeta,
                  
                       out_desc, dst));

           hipDeviceSynchronize();

           std::uint64_t time_elapsed = timer.elapsed_nanoseconds();

           time_vector[i] = (double)time_elapsed / 1000;

      }

	

    *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                                        time_vector.end(), 0) / (benchmark_iterations - 10);


  // finalizing

  hipdnnDestroyTensorDescriptor(out_desc);

  hipdnnDestroyTensorDescriptor(in_desc);

  hipdnnDestroyLRNDescriptor(lrn_desc);

  hipdnnDestroy(hipdnn);

}


#endif // TEST_LRN_FWD_COMMON_H
