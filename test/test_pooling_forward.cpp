#include "test_pooling_common.hpp"

hipdnnPoolingMode_t pool_mode;
hipdnnDataType_t dataType = HIPDNN_DATA_FLOAT;

TEST(pooling_fwd, func_check_zero_padding) {

  Desc inputDesc(1, 1, 4, 4);
  pool_mode = HIPDNN_POOLING_MAX;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0,0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 1.f;
  beta = 0.5f;

  float avg_time = 0;

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = srcData.getDataFromGPU();
  std::string ip = convert_to_string((float*)temp2,(int)srcData.get_num_elements());

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_zero_padding";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

TEST(pooling_fwd, func_check_AVERAGE_COUNT_INCLUDE_PADDING) {

  Desc inputDesc(1, 1, 4, 4);
  pool_mode = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {2, 2};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);
  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 2.f;
  beta = 0.f;

  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = srcData.getDataFromGPU();
  std::string ip = convert_to_string((float*)temp2,(int)srcData.get_num_elements());

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_AVERAGE_COUNT_INCLUDE_PADDING";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

TEST(pooling_fwd, func_check_batch32) {

  Desc inputDesc(32, 1, 224, 224);
  pool_mode = HIPDNN_POOLING_MAX;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);
  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 2.f;
  beta = 1.f;

  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_batch32";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

TEST(pooling_fwd, func_check_batch64) {

  Desc inputDesc(64, 1, 224, 224);
  pool_mode = HIPDNN_POOLING_MAX;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 2.f;
  beta = 0.5f;

  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_batch64";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

TEST(pooling_fwd, func_check_batch128) {

  Desc inputDesc(128, 1, 4, 4);
  pool_mode = HIPDNN_POOLING_MAX;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 1.f;
  beta = 0.f;

  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_batch128";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

TEST(pooling_fwd, func_check_rectangular_dims_small_size) {

  Desc inputDesc(1, 1, 4, 4);
  pool_mode = HIPDNN_POOLING_MAX;
  int spatial_ext[2] = {2, 1};
  int stride[2] = {2, 1};
  int pad[2] = {0,0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);
  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 1.f;
  beta = 0.f;

  float avg_time = 0;

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = srcData.getDataFromGPU();
  std::string ip = convert_to_string((float*)temp2,(int)srcData.get_num_elements());

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_rectangular_dims_small_size";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

TEST(pooling_fwd, func_check_rectangular_dims_medium_size) {

  Desc inputDesc(32, 1, 64, 64);
  pool_mode = HIPDNN_POOLING_MAX;
  int spatial_ext[2] = {4, 2};
  int stride[2] = {4, 2};
  int pad[2] = {0,0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 1.f;
  beta = 0.f;

  float avg_time = 0;

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = srcData.getDataFromGPU();
  std::string ip = convert_to_string((float*)temp2,(int)srcData.get_num_elements());

  float* temp = dstDataGPU.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_rectangular_dims_medium_size";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

/*
//half2

TEST(pooling_fwd, func_check_half) {

  Desc inputDesc(1, 1, 4, 4);
  pool_mode = HIPDNN_POOLING_MAX;
  dataType = HIPDNN_DATA_HALF;
  int spatial_ext[2] = {2, 2};
  int stride[2] = {2, 2};
  int pad[2] = {0, 0};

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);
  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  alpha = 1.f;
  beta = 0.f;

  float avg_time = 0;

  populateMemoryRandom<half>(srcData);
  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                         dataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  Memory<float> dstDataGPU_f(pool.mb * pool.c * pool.oh * pool.ow);

  Convert_toFloat<half>(dstDataGPU, dstDataGPU_f);
  float *temp = dstDataGPU_f.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_half_datatype";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
} */