#include "test_convolution_common.hpp"
#include "test_activation_common.hpp"

TEST(convolution_activation_fwd_bwd_intg,
     func_check_conv_activation_fwd_bwd) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0, avg_time3 = 0, avg_time4 = 0;

  Desc inputDesc(1, 3, 16, 16);
  Desc filterDesc(1, 3, 4, 4);
  hipdnnActivationMode_t act_mode = HIPDNN_ACTIVATION_RELU;

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {4, 4}; // stride 1
  int dil[2] = {1,1};

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride, dil);

  Memory<float> srcDataConv = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);
  Memory<float> gradData2 = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcDataConv);
  populateMemoryRandom<float>(filterData);

  convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], 1, 1);

  activation_params_t test_case(1, 1, 4, 4);

  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> data_grad(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  int ip_size_cf[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size_cf[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size_cf[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  int ip_size_af[4] = {test_case.n, test_case.channels, test_case.height,
                       test_case.width};
  int k_size_af[4] = {0,0,0,0};
  int op_size_af[4] = {test_case.n, test_case.channels, test_case.height,
                       test_case.width};

  int ip_size_ab[4] = {test_case.n, test_case.channels, test_case.height,
                       test_case.width};
  int k_size_ab[4] = {0,0,0,0};
  int op_size_ab[4] = {test_case.n, test_case.channels, test_case.height,
                       test_case.width};

  int ip_size_cb[4] = {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};
  int k_size_cb[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size_cb[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};

  std::string str_ip_size = integration_dims_to_string2(ip_size_cf,ip_size_af,
               ip_size_cb,ip_size_ab,"Conv_fwd","Act_fwd","Conv_bwd","Act_bwd");
  std::string str_k_size = integration_dims_to_string2(k_size_cf,k_size_af,
                 k_size_cb,k_size_ab,"Conv_fwd","Act_fwd","Conv_bwd","Act_bwd");
  std::string str_op_size = integration_dims_to_string2(op_size_cf,op_size_af,
               op_size_cb,op_size_ab,"Conv_fwd","Act_fwd","Conv_bwd","Act_bwd");

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcDataConv.gpu(),
                           filterData.gpu(), NULL, dstDataGPU.gpu(),&avg_time1);

  compute_hipdnn_activation_forward<float>(test_case, dstDataGPU.gpu(),
                                           dataDst.gpu(), act_mode, &avg_time2);

  compute_hipdnn_activation_backward<float>(test_case, dstDataGPU.gpu(),
                            data_grad.gpu(), dataDst.gpu(), act_mode, &avg_time3);

  compute_hipdnn_conv_backward_filter<float>(testConvolutionSizes, srcDataConv.gpu(),
                            filterData.gpu(), gradData2.gpu(), NULL,
                            data_grad.gpu(), &avg_time4);

  avg_time = avg_time1 + avg_time2 + avg_time3 + avg_time4;

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_activation_fwd_bwd_intg: func_check_conv_activation_fwd_bwd";
  std::string filename="convolution_activation_fwd_bwd.csv";

  float* temp = gradData2.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)gradData2.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData2.get_num_elements());

}
