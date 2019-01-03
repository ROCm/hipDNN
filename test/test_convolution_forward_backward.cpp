#include "test_convolution_common.hpp"

TEST(convolution_fwd_bwd, func_test_fwd_bwd_convolution) {

  Desc inputDesc(1, 3, 9, 9);
  Desc filterDesc(1, 3, 3, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

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

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcData.gpu(),
                           filterData.gpu(), NULL, dstDataGPU.gpu(), dataType,
                           &avg_time1);

  compute_hipdnn_conv_backward_filter<float>(testConvolutionSizes, srcData.gpu(),
                           filterData.gpu(), gradData.gpu(), NULL,
                           dstDataGPU.gpu(), dataType, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = gradData.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_intg:func_test_fwd_bwd_convolution";
  std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
  std::string filename="convolution_fwd_bwd_intg.csv";

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp2, (int)gradData.get_num_elements());
}

TEST(convolution_fwd_bwd, func_test_fwd_bwd_convolution_batch32) {

  Desc inputDesc(32, 3, 9, 9);
  Desc filterDesc(1, 3, 3, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

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

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), NULL, dstDataGPU.gpu(), dataType,
                                &avg_time1);
  compute_hipdnn_conv_backward_filter<float>(testConvolutionSizes, srcData.gpu(),
                                 filterData.gpu(), gradData.gpu(), NULL,
                                 dstDataGPU.gpu(), dataType, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = gradData.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_intg:func_test_fwd_bwd_convolution_batch32";
  std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
  std::string filename="convolution_fwd_bwd_intg.csv";

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp2, (int)gradData.get_num_elements());
}

TEST(convolution_fwd_bwd, func_test_fwd_bwd_convolution_batch64) {

  Desc inputDesc(64, 3, 9, 9);
  Desc filterDesc(21, 3, 3, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

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

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), NULL, dstDataGPU.gpu(), dataType,
                                &avg_time1);

  compute_hipdnn_conv_backward_filter<float>(testConvolutionSizes, srcData.gpu(),
                                 filterData.gpu(), gradData.gpu(), NULL,
                                 dstDataGPU.gpu(), dataType, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = gradData.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_intg:func_test_fwd_bwd_convolution_batch64";
  std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
  std::string filename="convolution_fwd_bwd_intg.csv";

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp2, (int)gradData.get_num_elements());
}

TEST(convolution_fwd_bwd, func_test_fwd_bwd_convolution_batch128) {

  Desc inputDesc(128, 3, 4, 4);
  Desc filterDesc(1, 3, 2, 2);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1,1};
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

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

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), NULL, dstDataGPU.gpu(), dataType,
                                &avg_time1);
  compute_hipdnn_conv_backward_filter<float>(testConvolutionSizes, srcData.gpu(),
                                 filterData.gpu(), gradData.gpu(), NULL,
                                 dstDataGPU.gpu(), dataType, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = gradData.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "convolution_intg:func_test_fwd_bwd_convolution_batch128";
  std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
  std::string filename="convolution_fwd_bwd_intg.csv";

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp2, (int)gradData.get_num_elements());
}