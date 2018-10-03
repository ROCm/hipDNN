#include "test_pooling_forward.hpp"

TEST(pooling_fwd, func_check_zero_padding) {
  test_2dpool_desc_t pool(1, 1, 224, 224, 224 / 2, 224 / 2, 2, 2, 0, 0, 2, 2);
  Memory<float> srcData(pool.ih * pool.iw);
  Memory<float> dstDataGPU((224 / 2) * (224 / 2));
  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu(), &avg_time);
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();
  
  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_zero_padding";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_fwd, func_check_batch32) {
  test_2dpool_desc_t pool(32, 1, 224, 224, 224 / 2, 224 / 2, 2, 2, 0, 0, 2, 2);
  Memory<float> srcData(pool.mb * pool.c * pool.ih * pool.iw);
  Memory<float> dstDataGPU(pool.mb * pool.c * pool.oh * pool.ow);
  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu(), &avg_time);
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();
  
  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_batch32";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_fwd, func_check_batch64) {
  test_2dpool_desc_t pool(64, 1, 224, 224, 224 / 2, 224 / 2, 2, 2, 0, 0, 2, 2);
  Memory<float> srcData(pool.mb * pool.c * pool.ih * pool.iw);
  Memory<float> dstDataGPU(pool.mb * pool.c * pool.oh * pool.ow);
  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu(), &avg_time);
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();
  
  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_batch64";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_fwd, func_check_batch128) {
  test_2dpool_desc_t pool(128, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  Memory<float> srcData(pool.mb * pool.c * pool.ih * pool.iw);
  Memory<float> dstDataGPU(pool.mb * pool.c * pool.oh * pool.ow);
  float avg_time = 0;

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu(), &avg_time);
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dstDataGPU.getDataFromGPU();
  
  std::string strt = "./result_unittest.csv";
  std::string testname = "pooling_fwd:func_check_batch128";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}
