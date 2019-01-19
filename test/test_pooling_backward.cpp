#include "test_pooling_common.hpp"

hipdnnPoolingMode_t poolB_mode;
hipdnnDataType_t dataType_b = HIPDNN_DATA_FLOAT;

TEST(pooling_backward, func_check_pooling_stride_2x2) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};
  float avg_time = 0;
  alpha = 1.f;
  beta = 0.5f;
  poolB_mode = HIPDNN_POOLING_MAX;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  test_pooling_descriptor test_case(inputDesc.N, inputDesc.C, inputDesc.H,
                          inputDesc.W, outputDesc.H, outputDesc.W, spatial_ext[0],
                          spatial_ext[1], pad[0], pad[1], stride[0], stride[1]);

  Memory<float> dataSrc = createMemory<float>(inputDesc);
  Memory<float> dataGrad = createMemory<float>(inputDesc);
  Memory<float> dataDst = createMemory<float>(outputDesc);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {inputDesc.N, inputDesc.C, spatial_ext[0], spatial_ext[1]};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                 dataDst.gpu(), poolB_mode, dataType_b, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dataGrad.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataGrad.get_num_elements());

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_backward:func_check_pooling_stride_2x2";
  std::string filename="pooling_backward.csv";

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataGrad.get_num_elements());
}

TEST(pooling_backward, func_check_pooling_AVERAGE_COUNT_INCLUDE_PADDING) {

  Desc inputDesc(1, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};
  float avg_time = 0;
  alpha = 1.f;
  beta = 0.5f;
  poolB_mode = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  test_pooling_descriptor test_case(inputDesc.N, inputDesc.C, inputDesc.H,
                          inputDesc.W, outputDesc.H, outputDesc.W, spatial_ext[0],
                          spatial_ext[1], pad[0], pad[1], stride[0], stride[1]);

  Memory<float> dataSrc = createMemory<float>(inputDesc);
  Memory<float> dataGrad = createMemory<float>(inputDesc);
  Memory<float> dataDst = createMemory<float>(outputDesc);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {inputDesc.N, inputDesc.C, spatial_ext[0], spatial_ext[1]};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(), dataDst.gpu(),
                                 poolB_mode, dataType_b, alpha, beta,&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dataGrad.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataGrad.get_num_elements());

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_backward:func_check_pooling_AVERAGE_COUNT_INCLUDE_PADDING";
  std::string filename="pooling_backward.csv";

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataGrad.get_num_elements());
}

TEST(pooling_backward, func_check_pooling_batch32) {

  Desc inputDesc(32, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};
  float avg_time = 0;
  alpha = 2.f;
  beta = 0.f;
  poolB_mode = HIPDNN_POOLING_MAX;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  test_pooling_descriptor test_case(inputDesc.N, inputDesc.C, inputDesc.H,
                          inputDesc.W, outputDesc.H, outputDesc.W, spatial_ext[0],
                          spatial_ext[1], pad[0], pad[1], stride[0], stride[1]);

  Memory<float> dataSrc = createMemory<float>(inputDesc);
  Memory<float> dataGrad = createMemory<float>(inputDesc);
  Memory<float> dataDst = createMemory<float>(outputDesc);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {inputDesc.N, inputDesc.C, spatial_ext[0], spatial_ext[1]};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                 dataDst.gpu(), poolB_mode, dataType_b, alpha, beta, &avg_time);

  float* temp = dataGrad.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataGrad.get_num_elements());

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_backward:func_check_pooling_batch32";
  std::string filename="pooling_backward.csv";

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataGrad.get_num_elements());
}

TEST(pooling_backward, func_check_pooling_batch64) {

  Desc inputDesc(64, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};
  float avg_time = 0;
  alpha = 0.5f;
  beta = 0.f;
  poolB_mode = HIPDNN_POOLING_MAX;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

    test_pooling_descriptor test_case(inputDesc.N, inputDesc.C, inputDesc.H,
                          inputDesc.W, outputDesc.H, outputDesc.W, spatial_ext[0],
                          spatial_ext[1], pad[0], pad[1], stride[0], stride[1]);

  Memory<float> dataSrc = createMemory<float>(inputDesc);
  Memory<float> dataGrad = createMemory<float>(inputDesc);
  Memory<float> dataDst = createMemory<float>(outputDesc);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {inputDesc.N, inputDesc.C, spatial_ext[0], spatial_ext[1]};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                 dataDst.gpu(), poolB_mode, dataType_b, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_backward:func_check_pooling_batch64";
  std::string filename="pooling_backward.csv";

  float* temp = dataGrad.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());
  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataGrad.get_num_elements());
}

TEST(pooling_backward, func_check_pooling_batch128) {

  Desc inputDesc(128, 1, 4, 4);
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};
  float avg_time = 0;
  alpha = 1.f;
  beta = 0.f;
  poolB_mode = HIPDNN_POOLING_MAX;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

    test_pooling_descriptor test_case(inputDesc.N, inputDesc.C, inputDesc.H,
                          inputDesc.W, outputDesc.H, outputDesc.W, spatial_ext[0],
                          spatial_ext[1], pad[0], pad[1], stride[0], stride[1]);

  Memory<float> dataSrc = createMemory<float>(inputDesc);
  Memory<float> dataGrad = createMemory<float>(inputDesc);
  Memory<float> dataDst = createMemory<float>(outputDesc);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {inputDesc.N, inputDesc.C, spatial_ext[0], spatial_ext[1]};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                 dataDst.gpu(), poolB_mode, dataType_b, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_backward:func_check_pooling_batch128";
  std::string filename="pooling_backward.csv";

  float* temp = dataGrad.getDataFromGPU();
  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataGrad.get_num_elements());
}