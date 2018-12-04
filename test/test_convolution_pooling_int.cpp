#include "test_convolution_pooling_int.hpp"

TEST(convolution_pooling_fwd_intg, func_check_naive_conv_pool) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;
  int oheight = 4, owidth = 4;

  test_pooling_descriptor pool(1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);

  Memory<float> dstData(pool.mb * pool.c * pool.oh * pool.ow);

  Desc inputDesc(1, 3, 16, 16);
  Desc filterDesc(1, 3, 4, 4);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {4, 4}; // stride 1
  int dil[2] = {1,1};

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

  compute_conv_forward<float>(testConvolutionSizes, srcDataConv.gpu(),
                           filterData.gpu(), NULL, dstDataGPU.gpu(),&avg_time1);

  compute_pooling_fwd<float>(pool, dstDataGPU.gpu(), dstData.gpu(), &avg_time2);

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