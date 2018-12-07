#include "test_convolution_backward_data.hpp"

TEST(convolution_bwd_data, func_check_backward_conv_data) {

  Desc inputDesc(1, 3, 30, 30);
  Desc filterDesc(3, 3, 5, 5);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {5, 5}; // stride 1
  float avg_time = 0;
  int dil[2] = {1,1};

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size conv_back_param(
    inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
    outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
    stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_conv_backward_data<float>(conv_back_param, srcData.gpu(),
           filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu(), &avg_time);

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_data:func_check_backward_conv_img";
  std::string filename="convolution_bwd_data.csv";

  float* temp = gradData.getDataFromGPU();
  std::string str  = convert_to_string((float*)temp,
                                       (int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_img, func_bwd_conv_batch8) {

  Desc inputDesc(8, 3, 3, 3);
  Desc filterDesc(8, 3, 2, 2);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};
  float avg_time = 0;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size conv_back_param(
    inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
    outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
    stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_conv_backward_data<float>(conv_back_param, srcData.gpu(),
           filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu(), &avg_time);

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_data:func_bwd_conv_batch8";
  std::string filename="convolution_bwd_data.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_img, func_bwd_conv_batch16) {

  Desc inputDesc(16, 3, 3, 3);
  Desc filterDesc(16, 3, 2, 2);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};

  float avg_time = 0;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size conv_back_param(
    inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
    outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
    stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_conv_backward_data<float>(conv_back_param, srcData.gpu(),
           filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu(), &avg_time);

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_data:func_bwd_conv_batch16";
  std::string filename="convolution_bwd_data.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_img, func_bwd_conv_batch64_pad) {

  Desc inputDesc(64, 3, 3, 3);
  Desc filterDesc(64, 3, 2, 2);

  int pad[2] = {0, 0};
  int stride[2] = {1, 1};
  float avg_time = 0;
  int dil[2] = {1,1};

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size conv_back_param(
    inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
    outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
    stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_conv_backward_data<float>(conv_back_param, srcData.gpu(),
          filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu(), &avg_time);

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_img:func_bwd_conv_batch64_pad";
  std::string filename="convolution_bwd_data.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}

TEST(convolution_bwd_img, func_bwd_conv_batch128_pad1_stride3) {

  Desc inputDesc(128, 3, 3, 3);
  Desc filterDesc(128, 3, 2, 2);

  int pad[2] = {0, 0};
  int stride[2] = {1, 1};
  int dil[2] = {1,1};
  float avg_time = 0;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size conv_back_param(
    inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
    outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
    stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_conv_backward_data<float>(conv_back_param, srcData.gpu(),
           filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu(), &avg_time);

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_bwd_data:func_bwd_conv_batch128_pad1_stride3";
  std::string filename="convolution_bwd_data.csv";
  float* temp = gradData.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradData.get_num_elements());
}