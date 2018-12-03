#include "test_LRN_forward.hpp"

TEST(LRN_fwd, func_check_naive_LRN) {

  Desc inputDesc(1, 3, 7, 7);
  float avg_time = 0;

  Desc outputDesc(1, 3, 7, 7);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "LRN_fwd:func_check_naive_LRN";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

}

TEST(LRN_fwd, func_check_LRN_batch16) {

  Desc inputDesc(16, 3, 60, 60);
  float avg_time = 0;

  Desc outputDesc(16, 3, 60, 60);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "LRN_fwd:func_check_LRN_batch16";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

}

TEST(LRN_fwd, func_check_LRN_batch32) {

  Desc inputDesc(16, 3, 60, 60);
  float avg_time = 0;

  Desc outputDesc(16, 3, 60, 60);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  populateMemoryRandom<float>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "LRN_fwd:func_check_LRN_batch32";

  float* temp = dstDataGPU.getDataFromGPU();
  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

}

TEST(LRN_fwd, func_check_LRN_batch64) {

  Desc inputDesc(64, 3, 7, 7);
  float avg_time = 0;

  Desc outputDesc(64, 3, 7, 7);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "LRN_fwd:func_check_LRN_batch64";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

}

TEST(LRN_fwd, func_check_LRN_batch128) {

  Desc inputDesc(128, 3, 7, 7);
  float avg_time = 0;

  Desc outputDesc(128, 3, 7, 7);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(LRN_params, srcData.gpu(), dstDataGPU.gpu(),
                                &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "LRN_fwd:func_check_LRN_batch128";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

}