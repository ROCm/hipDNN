#include "test_pooling_common.hpp"
#include "test_convolution_common.hpp"

hipdnnPoolingMode_t poolCF_mode;

TEST(convolution_pooling_fwd_intg, func_check_naive_conv_pool) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;
  Desc inputDescP(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int strideP[2] = {2, 2};
  int pad_p[2] = {0,0};
  poolCF_mode = HIPDNN_POOLING_MAX;
  hipdnnDataType_t dataType = HIPDNN_DATA_FLOAT;

  Desc outputDescP = calculate_pool_Dims(inputDescP, spatial_ext, pad_p, strideP);

  test_pooling_descriptor pool(inputDescP.N, inputDescP.C, inputDescP.H,
                               inputDescP.W, outputDescP.H, outputDescP.W,
                               spatial_ext[0], spatial_ext[1], pad_p[0], pad_p[1],
                               strideP[0], strideP[1]);

  Memory<float> dstData = createMemory<float>(outputDescP);

  Desc inputDesc(1, 3, 16, 16);
  Desc filterDesc(1, 3, 4, 4);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {4, 4}; // stride 1
  int dil[2] = {1,1};
  alpha = 1.f;
  beta = 0.f;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride, dil);

  Memory<float> srcDataConv = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcDataConv);
  populateMemoryRandom<float>(filterData);

  convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size_c[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size_c[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size_c[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  int ip_size_p[4] = {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};
  int k_size_p[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size_p[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size = integration_dims_to_string(ip_size_c,ip_size_p,
                                                      "Conv","MP");
  std::string str_k_size = integration_dims_to_string(k_size_c,k_size_p,
                                                      "Conv","MP");
  std::string str_op_size = integration_dims_to_string(op_size_c,op_size_p,
                                                       "Conv","MP");

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcDataConv.gpu(),
                           filterData.gpu(), NULL, dstDataGPU.gpu(), &alpha,
                           &beta, &avg_time1);

  hipdnn_pooling_forward<float>(pool, dstDataGPU.gpu(), dstData.gpu(), poolCF_mode,
                                dataType, false, alpha, beta, &avg_time2);

  avg_time = avg_time1 + avg_time2;

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_pooling_fwd_intg:func_check_naive_conv_pool";
  std::string filename="convolution_pooling_fwd_intg.csv";

  float* temp = dstData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)dstData.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstData.get_num_elements());
}
