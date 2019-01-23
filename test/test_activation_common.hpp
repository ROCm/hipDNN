#ifndef TEST_ACTIVATION_COMMON_HPP
#define TEST_ACTIVATION_COMMON_HPP

#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_activation_forward(activation_params_t &test_case,
                                        dataType *src,
                                        dataType *dst,
                                        hipdnnActivationMode_t mode,
                                        float alpha, float beta,
                                        float *avg_time) {
  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));

  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));

  hipdnnActivationDescriptor_t activationDesc;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;

  double reluCeilingOrAlpha = 1;
  double activBeta = 1;
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

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        hipdnnActivationForward(hipdnn, activationDesc, &alpha, in_desc, src,
                               &beta, out_desc, dst);
        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                                     time_vector.end(), 0)
                                     / (benchmark_iterations - 10);

  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyActivationDescriptor(activationDesc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

template <typename dataType>
void compute_hipdnn_activation_backward(activation_params_t &test_case,
                                        dataType *src, dataType *grad,
                                        dataType *dst,
                                        hipdnnActivationMode_t mode,
                                        float alpha, float beta,
                                        float *avg_time) {
  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));

  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));

  hipdnnActivationDescriptor_t activationDesc;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;

  double reluCeilingOrAlpha = 1;
  double activBeta = 1;
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

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

      timer.restart();

      hipdnnActivationBackward(hipdnn, activationDesc, &alpha, in_desc, src,
                           in_desc, src, out_desc, dst, &beta, out_desc, grad);

      hipDeviceSynchronize();

      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                    0) / (benchmark_iterations - 10);

  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyActivationDescriptor(activationDesc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void Test_activation_fwd(activation_params_t test_case,
                         hipdnnActivationMode_t act_mode, std::string testname,
                         float alpha, float beta)
{
  float avg_time = 0;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    act_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dataDst.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string filename = "activation_forward.csv";

  std::string str  = convert_to_string((float*)temp,
                                         (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
                 str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());
}

template <typename dataType>
void Test_activation_bwd(activation_params_t test_case,
                         hipdnnActivationMode_t act_mode, std::string testname,
                         float alpha, float beta)
{
  float avg_time = 0;

  Memory<dataType> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<dataType> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<dataType> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    act_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<dataType>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            act_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());
}

#endif //TEST_ACTIVATION_FORWARD_HPP
