#include "test_convolution_backward_filter.hpp"

TEST(convolution_bwd_filter, func_check_backward_conv_filter) {

  Desc inputDesc(1, 3, 30, 30);
  Desc filterDesc(1, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  float avg_time = 0;
  int dil[2] = {1,1};

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(filterDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  convulution_Size testConvolutionSizes(
      inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
      outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
      stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  compute_hipdnn_conv_bwd_filter<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), gradData.gpu(), NULL,
                                dstDataGPU.gpu(),&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_filter:func_check_backward_conv_filter";
  std::string filename="convolution_bwd_filter.csv";

  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_filter, func_backward_conv_filter_batch32) {

  Desc inputDesc(32, 3, 10, 10);
  Desc filterDesc(32, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};
  float avg_time = 0;

  Desc outputDesc =
    calculate_Dims(inputDesc, filterDesc, pad, stride, dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(filterDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  convulution_Size testConvolutionSizes(
      inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
      outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
      stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  compute_hipdnn_conv_bwd_filter<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), gradData.gpu(),
                                NULL, dstDataGPU.gpu(),&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_filter:func_backward_conv_filter_batch32";
  std::string filename="convolution_bwd_filter.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());
  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_filter, func_backward_conv_filter_batch64) {

  Desc inputDesc(64, 3, 10, 10);
  Desc filterDesc(64, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};
  float avg_time = 0;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride, dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(filterDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  convulution_Size testConvolutionSizes(
      inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
      outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
      stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  compute_hipdnn_conv_bwd_filter<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), gradData.gpu(),
                                NULL, dstDataGPU.gpu(),&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_filter:func_backward_conv_filter_batch64";
  std::string filename="convolution_bwd_filter.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_filter, func_backward_conv_filter_batch128) {

  Desc inputDesc(128, 3, 30, 30);
  Desc filterDesc(3, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 5
  int dil[2] = {1,1};
  float avg_time = 0;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride, dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(filterDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  convulution_Size testConvolutionSizes(
    inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
    outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
    stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  compute_hipdnn_conv_bwd_filter<float>(testConvolutionSizes, srcData.gpu(),
            filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu(),&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_filter:func_backward_conv_filter_batch128";
  std::string filename="convolution_bwd_filter.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}