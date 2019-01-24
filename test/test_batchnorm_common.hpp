#ifndef TEST_BATCHNORM_COMMON_HPP
#define TEST_BATCHNORM_COMMON_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "gtest/gtest.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_batchnorm_fwd_train(BNorm_params_t &d, dataType *src,
                              dataType *dst, dataType *resultRunningMean,
                              dataType *resultRunningVariance,
                              dataType *resultSaveMean,
                              dataType *resultSaveInvVariance,
                              float *avg_time, hipdnnBatchNormMode_t bn_modeT,
                              hipdnnDataType_t hipdataType) {

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

  hipdnnTensorDescriptor_t bnScaleBiasMeanVarDescT;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDescT));
  checkHIPDNN(hipdnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDescT, out_desc,
                                             bn_modeT));

  checkHIPDNN(hipdnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDescT,
                                             in_desc, bn_modeT));

  int out_n,out_c,out_h,out_w, nStride,cStride,hStride,wStride;

  hipdnnDataType_t dt = hipdataType;

  hipdnnGetTensor4dDescriptor(
             bnScaleBiasMeanVarDescT, &dt, &out_n, &out_c, &out_h, &out_w, &nStride,
             &cStride, &hStride, &wStride);

  float* bnScaleT;
  HIP_CALL(hipMalloc(&bnScaleT, out_h * out_w * out_c * sizeof(float)));
  hipLaunchKernel(dev_const, out_w * out_h, out_c, 0, 0 ,bnScaleT ,1.f);

  float* bnBiasT;
  HIP_CALL(hipMalloc(&bnBiasT, out_h * out_w * out_c * sizeof(float)));
  hipLaunchKernel(dev_const, out_w * out_h, out_c, 0, 0 ,bnBiasT ,0.f);

  float alphaNT = 1.f;
  float betaNT = 0.f;
  double epsilonT = 2.f;
  double exponentialAverageFactor = 0.5;

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

      timer.restart();

         checkHIPDNN( hipdnnBatchNormalizationForwardTraining( hipdnn, bn_modeT,
                              &alphaNT, &betaNT, in_desc, src, out_desc, dst,
                              bnScaleBiasMeanVarDescT, bnScaleT, bnBiasT,
                              exponentialAverageFactor, resultRunningMean,
                              resultRunningVariance, epsilonT,
                              resultSaveMean, resultSaveInvVariance));

     hipDeviceSynchronize();

     std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
     time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                                     time_vector.end(), 0)
                                     / (benchmark_iterations - 10);

  // finalizing
  hipFree(bnScaleT);
  hipFree(bnBiasT);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasMeanVarDescT);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void compute_hipdnn_batchnorm_fwd_inference(BNorm_params_t &d, dataType *src,
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

  hipdnnBatchNormMode_t bn_mode = HIPDNN_BATCHNORM_SPATIAL;
    /* HIPDNN_BATCHNORM_SPATIAL_PERSISTENT*/
  float alphaN = 1.f;
  float betaN = 0.f;

  hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  /*checkHIPDNN(hipdnnDeriveBNTensorDescriptor(&bnScaleBiasMeanVarDesc,
                         out_desc, bn_mode));*/           //Only for training
  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
          bnScaleBiasMeanVarDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
          1, d.ic, 1, 1));

  float* bnScale;
  HIP_CALL(hipMalloc(&bnScale, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,bnScale ,1.f);

  float* bnBias;
  HIP_CALL(hipMalloc(&bnBias, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,bnBias ,0.f);

  float* estimatedMean;
  HIP_CALL(hipMalloc(&estimatedMean, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,estimatedMean ,0.f);

  float* estimatedVariance;
  HIP_CALL(hipMalloc(&estimatedVariance, 1*d.ic*1*1*sizeof(float)));
  hipLaunchKernel(dev_const, 1*d.ic, 1*1, 0, 0 ,estimatedVariance ,2.f);

  double epsilon = 2.f; // Minimum 1e-5

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        checkHIPDNN(hipdnnnBatchNormalizationForwardInference( hipdnn, bn_mode,
                    &alphaN, &betaN, out_desc, src, out_desc , dst,
                    bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
                    estimatedVariance, epsilon));

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
      }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                     0) / (benchmark_iterations - 10);

  // finalizing
  hipFree(bnScale);
  hipFree(bnBias);
  hipFree(estimatedMean);
  hipFree(estimatedVariance);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void compute_hipdnn_batchnorm_bwd(BNorm_params_t &d, dataType *src,
                                dataType *dx, dataType *resultBnScaleDiff,
                                dataType *resultBnBiasDiff, float *avg_time,
                                hipdnnBatchNormMode_t bn_modeT_back, int acc_grad,
                                hipdnnDataType_t hipdataType) {

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

  hipdnnTensorDescriptor_t bnScaleBiasDiffDesc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&bnScaleBiasDiffDesc));

  checkHIPDNN(hipdnnDeriveBNTensorDescriptor(bnScaleBiasDiffDesc,
                                             in_desc, bn_modeT_back));

  int out_n,out_c,out_h,out_w, nStride,cStride,hStride,wStride;

  hipdnnDataType_t dt = hipdataType;

  hipdnnGetTensor4dDescriptor(
             bnScaleBiasDiffDesc, &dt, &out_n, &out_c, &out_h, &out_w, &nStride,
             &cStride, &hStride, &wStride);

  hipdnnTensorDescriptor_t dy_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&dy_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
              dy_desc, HIPDNN_TENSOR_NCHW, hipdataType,
              d.mb, d.ic, d.ih, d.iw));

  float* dy;

  HIP_CALL(hipMalloc(&dy, d.mb*d.ic*d.ih*d.iw*sizeof(float)));

  hipLaunchKernel(dev_iota, d.iw * d.ih, d.mb * d.ic , 0, 0 ,dy);

  hipdnnTensorDescriptor_t dx_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&dx_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
              dx_desc, HIPDNN_TENSOR_NCHW, hipdataType,
              d.mb, d.ic, d.ih, d.iw));

  float* bnScaleT_back;
  HIP_CALL(hipMalloc(&bnScaleT_back, out_h * out_w * out_c * sizeof(float)));
  hipLaunchKernel(dev_const, out_w * out_h, out_c , 0, 0 ,bnScaleT_back ,1.f);

  float alphaDataDiff = 1.f;
  float betaDataDiff = 0.f;
  float alphaParamDiff = 1.f;
  float betaParamDiff = 0.f;
  double epsilonT_back = 1e-5;

  float *savedMean;

  hipMalloc(
        &savedMean, out_h * out_w * out_c * sizeof(float));

  hipLaunchKernel(dev_iota, out_w * out_h,  out_c , 0, 0 ,savedMean);

  float *savedInvVariance;
  hipMalloc(
        &savedInvVariance, out_h * out_w * out_c * sizeof(float));
  hipLaunchKernel(dev_iota, out_w * out_h,  out_c , 0, 0 ,savedInvVariance);

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

              checkHIPDNN( hipdnnBatchNormalizationBackward( hipdnn, bn_modeT_back,
                      &alphaDataDiff, &betaDataDiff, &alphaParamDiff,
                      &betaParamDiff, in_desc, src, dy_desc, dy, dx_desc, dx,
                      bnScaleBiasDiffDesc, bnScaleT_back, resultBnScaleDiff,
                      resultBnBiasDiff, epsilonT_back, savedMean,
                      savedInvVariance));

        if (acc_grad == 1)
           {
              betaDataDiff = 1.f;
              betaParamDiff = 1.f;

              checkHIPDNN( hipdnnBatchNormalizationBackward( hipdnn,
                            bn_modeT_back, &alphaDataDiff, &betaDataDiff,
                            &alphaParamDiff, &betaParamDiff, in_desc, src,
                            dy_desc, dy, dx_desc, dx, bnScaleBiasDiffDesc,
                            bnScaleT_back, resultBnScaleDiff, resultBnBiasDiff,
                            epsilonT_back, savedMean, savedInvVariance));
           }

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;

      }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                            time_vector.end(), 0) / (benchmark_iterations - 10);

  hipFree(bnScaleT_back);
  hipFree(savedMean);
  hipFree(savedInvVariance);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroyTensorDescriptor(dy_desc);
  hipdnnDestroyTensorDescriptor(dx_desc);
  hipdnnDestroyTensorDescriptor(bnScaleBiasDiffDesc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void Test_bnorm_fwd_train(Desc inputDesc, Desc outputDesc,
                          Desc bnScaleBiasMeanVarDesc,
                          hipdnnBatchNormMode_t bn_mode, std::string testname[4],
                          hipdnnDataType_t hipdataType = HIPDNN_DATA_FLOAT){

  float avg_time = 0;

  Memory<dataType> srcData = createMemory<dataType>(inputDesc);
  Memory<dataType> dstDataGPU = createMemory<dataType>(outputDesc);
  Memory<dataType> resultRunningMean = createMemory<dataType>(bnScaleBiasMeanVarDesc);
  Memory<dataType> resultRunningVariance = createMemory<dataType>(bnScaleBiasMeanVarDesc);
  Memory<dataType> resultSaveMean = createMemory<dataType>(bnScaleBiasMeanVarDesc);
  Memory<dataType> resultSaveVariance = createMemory<dataType>(bnScaleBiasMeanVarDesc);

  populateMemoryRandom<dataType>(srcData);
  populateMemoryRandom<dataType>(resultRunningMean);
  populateMemoryRandom<dataType>(resultRunningVariance);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  compute_hipdnn_batchnorm_fwd_train<dataType>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(),
                                            resultRunningMean.gpu(),
                                            resultRunningVariance.gpu(),
                                            resultSaveMean.gpu(),
                                            resultSaveVariance.gpu(), &avg_time,
                                            bn_mode, hipdataType);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string filename="BNorm_Fwd_train.csv";

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);


  float* temp = dstDataGPU.getDataFromGPU();
  float* temp1 = resultRunningMean.getDataFromGPU();
  float* temp2 = resultRunningVariance.getDataFromGPU();
  float* temp3 = resultSaveMean.getDataFromGPU();
  float* temp4 = resultSaveVariance.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());
  std::string str1  = convert_to_string((float*)temp1,
                                       (int)resultRunningMean.get_num_elements());
  std::string str2  = convert_to_string((float*)temp2,
                                       (int)resultRunningVariance.get_num_elements());
  std::string str3  = convert_to_string((float*)temp3,
                                       (int)resultSaveMean.get_num_elements());
  std::string str4  = convert_to_string((float*)temp4,
                                       (int)resultSaveVariance.get_num_elements());

  write_to_csv(strt, str, testname[0], avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str1, testname[1], avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str2, testname[2], avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str3, testname[3], avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str4, testname[4], avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname[0], temp, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname[1], temp1, (int)resultRunningMean.get_num_elements());
  dump_result_csv(filename, testname[2], temp2, (int)resultRunningVariance.get_num_elements());
  dump_result_csv(filename, testname[3], temp3, (int)resultSaveMean.get_num_elements());
  dump_result_csv(filename, testname[4], temp4, (int)resultSaveVariance.get_num_elements());

}

template <typename dataType>
void Test_bnorm_bwd(Desc inputDesc, Desc outputDesc,
            hipdnnBatchNormMode_t bn_mode, int acc_grad, std::string testname[2],
            hipdnnDataType_t hipdataType = HIPDNN_DATA_FLOAT){

  float avg_time = 0;

  Memory<dataType> srcData = createMemory<dataType>(inputDesc);
  Memory<dataType> dstDataGPU = createMemory<dataType>(inputDesc);
  Memory<dataType> resultBnScaleDiff = createMemory<dataType>(outputDesc);
  Memory<dataType> resultBnBiasDiff = createMemory<dataType>(outputDesc);

  populateMemoryRandom<dataType>(srcData);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_bwd<dataType>(BN_sizes, srcData.gpu(), dstDataGPU.gpu(),
                              resultBnScaleDiff.gpu(), resultBnBiasDiff.gpu(),
                              &avg_time, bn_mode, acc_grad, hipdataType);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string filename="BNorm_backward.csv";

  float* temp1 = dstDataGPU.getDataFromGPU();
  float* temp2 = resultBnScaleDiff.getDataFromGPU();
  float* temp3 = resultBnBiasDiff.getDataFromGPU();

  std::string str1  = convert_to_string2((float*)temp1,
                                    (int)dstDataGPU.get_num_elements());
  std::string str2  = convert_to_string2((float*)temp2,
                                    (int)resultBnScaleDiff.get_num_elements());
  std::string str3  = convert_to_string2((float*)temp3,
                                    (int)resultBnBiasDiff.get_num_elements());

  write_to_csv(strt, str1, testname[0], avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str2, testname[1], avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str3, testname[2], avg_time, str_ip_size, str_k_size,
              str_op_size);

  dump_result_csv(filename, testname[0], temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname[1], temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname[2], temp3, (int)resultBnBiasDiff.get_num_elements());

}

#endif //TEST_BATCH_NORM_BWD_HPP