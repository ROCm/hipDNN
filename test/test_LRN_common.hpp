#ifndef TEST_LRN_COMMON_HPP
#define TEST_LRN_COMMON_HPP

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

#define maxvalue 17
//Maxvalue = half? 5 : 17
//TO:DO Set max value wrt to datatype

template <typename dataType>
void compute_hipdnn_LRN_fwd(LRN_params_t &d, dataType *src, dataType *dst,
                            float *avg_time, hipdnnDataType_t hipdataType) {

  hipdnnHandle_t hipdnn;
	checkHIPDNN(hipdnnCreate(&hipdnn));

	hipdnnTensorDescriptor_t in_desc;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(in_desc, HIPDNN_TENSOR_NCHW,
                                          hipdataType, d.mb, d.ic, d.ih,
                                          d.iw));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(out_desc, HIPDNN_TENSOR_NCHW,
                                          hipdataType, d.mb, d.ic, d.ih,
                                          d.iw));

  hipdnnLRNDescriptor_t lrn_desc;
  hipdnnCreateLRNDescriptor(&lrn_desc);

  hipdnnLRNMode_t lrn_mode = HIPDNN_LRN_CROSS_CHANNEL;

  unsigned lrn_n = 5 ; // cudnn default 5  (at desc creation)
  double lrn_alpha = 0.01 ; // cudnn default 1e-4
  double lrn_beta = 0.5 ; // cudnn default 0.75
  double lrn_k = 1.0 ;  // cudnn default 2.0
  checkHIPDNN(hipdnnSetLRNDescriptor(lrn_desc, lrn_mode, lrn_n, lrn_alpha,
                                     lrn_beta, lrn_k));

  float lrn_blendAlpha = 1.f;
  float lrn_blendBeta = 0.5;
  bool do_backward = false;

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

         timer.restart();
         checkHIPDNN(hipdnnLRNCrossChannelForward( hipdnn, lrn_desc, lrn_mode,
                 &lrn_blendAlpha, in_desc, src, &lrn_blendBeta, out_desc, dst,
                 do_backward));

         hipDeviceSynchronize();

         std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
         time_vector[i] = (double)time_elapsed / 1000;
	}

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                    0) / (benchmark_iterations - 10);

  // finalizing
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyLRNDescriptor(lrn_desc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void compute_hipdnn_LRN_backward(LRN_params_t &d, dataType *src, dataType *grad,
                             dataType *dst, float *avg_time,
                             hipdnnDataType_t hipdataType = HIPDNN_DATA_FLOAT) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        in_desc, HIPDNN_TENSOR_NCHW, hipdataType,
        d.mb, d.ic, d.ih, d.iw));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
        out_desc, HIPDNN_TENSOR_NCHW, hipdataType,
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
                &lrn_blendAlpha, in_desc, src, &lrn_blendBeta, out_desc, dst, true));


  float* dy; // passed as input
  HIP_CALL(hipMalloc(&dy, d.mb*d.ic*d.ih*d.iw*sizeof(float)));
  hipLaunchKernel(dev_populate, d.ih*d.iw, d.mb*d.ic, 0, 0 , dy, maxvalue);


  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        checkHIPDNN(hipdnnLRNCrossChannelBackward( hipdnn, lrn_desc, lrn_mode,
                   &lrn_blendAlpha, out_desc, dst, out_desc, dy, in_desc, src,
                   &lrn_blendBeta, out_desc, grad));

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
	}

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                            time_vector.end(), 0) / (benchmark_iterations - 10);


  // finalizing
  hipFree(dy);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyLRNDescriptor(lrn_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void Test_LRN_fwd(Desc inputDesc, Desc outputDesc, std::string testname,
                  hipdnnDataType_t hipdataType = HIPDNN_DATA_FLOAT) {

  float avg_time = 0;
  dataType* temp;

  Memory<dataType> srcData = createMemory<dataType>(inputDesc);
  Memory<dataType> dstDataGPU = createMemory<dataType>(outputDesc);

  populateMemoryRandom<dataType>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<dataType>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time, hipdataType);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string filename="LRN_fwd.csv";

    temp =  dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}


template <>
inline void Test_LRN_fwd<half>(Desc inputDesc, Desc outputDesc, std::string testname,
                  hipdnnDataType_t hipdataType) {

  float avg_time = 0;
  float* temp;
  hipdataType = HIPDNN_DATA_FLOAT;

  Memory<half> srcData = createMemory<half>(inputDesc);
  Memory<half> dstDataGPU = createMemory<half>(outputDesc);

  populateMemoryRandom<half>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<half>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time, hipdataType);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string filename="LRN_fwd.csv";


    Memory<float> dstDataGPU_f(outputDesc.N * outputDesc.C * outputDesc.H * outputDesc.W);
    Convert_toFloat<half>(dstDataGPU, dstDataGPU_f);
    temp = dstDataGPU_f.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

#endif // TEST_LRN_FWD_COMMON_H
