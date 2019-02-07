#include "test_convolution_common.hpp"
#include "test_activation_common.hpp"

TEST(convolution_activation_fwd_intg, func_check_naive_conv_activation) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

  Desc inputDesc(1, 3, 16, 16);
  Desc filterDesc(1, 3, 4, 4);
  hipdnnActivationMode_t act_mode = HIPDNN_ACTIVATION_RELU;
  hipdnnDataType_t dataType = HIPDNN_DATA_FLOAT;

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {4, 4}; // stride 1
  int dil[2] = {1,1};
  alpha = 1.f;
  beta = 0.f;

  Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);

  Memory<float> srcDataConv = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcDataConv);
  populateMemoryRandom<float>(filterData);
  convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], 1, 1);

  activation_params_t test_case(1, 1, 4, 4);

  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  int ip_size_c[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size_c[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size_c[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  int ip_size_a[4] = {test_case.n, test_case.channels, test_case.height,
                      test_case.width};
  int k_size_a[4] = {0,0,0,0};
  int op_size_a[4] =  {test_case.n, test_case.channels, test_case.height,
                       test_case.width};

  std::string str_ip_size = integration_dims_to_string(ip_size_c,ip_size_a,
                                                       "Conv","Act");
  std::string str_k_size = integration_dims_to_string(k_size_c,k_size_a,
                                                      "Conv","Act");
  std::string str_op_size = integration_dims_to_string(op_size_c,op_size_a,
                                                       "Conv","Act");

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcDataConv.gpu(),
                           filterData.gpu(), NULL, dstDataGPU.gpu(), alpha,
                           beta, &avg_time1);

  compute_hipdnn_activation_forward<float>(test_case, dstDataGPU.gpu(),
                                           dataDst.gpu(), dataType, act_mode,
                                           alpha, beta, &avg_time2);

  avg_time = avg_time1 + avg_time2;

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_activation_fwd_intg: func_check_naive_conv_activation";
  std::string filename="convolution_activation_fwd_intg.csv";

  float* temp = dataDst.getDataFromGPU();
  std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());

}