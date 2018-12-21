#include "test_batch_norm_fwd_training.hpp"
int modeB; //0:SPATIAL 1:Per_Activation

TEST(BNorm_Fwd_train, func_check_spatial_fwd) {
  Desc inputDesc(1, 1, 4, 4);
  Desc outputDesc(1, 1, 1, 1);

  float avg_time = 0;
  modeB = 0;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(), &avg_time, modeB);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_spatial_fwd";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());

}

TEST(BNorm_Fwd_train, func_check_per_act_fwd) {
  Desc inputDesc(1, 1, 4, 4);
  Desc outputDesc(1, 1, 4, 4);

  float avg_time = 0;
  modeB = 0;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(), &avg_time, modeB);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_per_act_fwd";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());

}

TEST(BNorm_Fwd_train, func_check_spatial_fwd_channel3) {
  Desc inputDesc(1, 3, 4, 4);
  Desc outputDesc(1, 3, 1, 1);

  float avg_time = 0;
  modeB = 0;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(), &avg_time, modeB);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_spatial_fwd_channel3";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());

}

TEST(BNorm_Fwd_train, func_check_per_act_fwd_channel3) {
  Desc inputDesc(1, 3, 4, 4);
  Desc outputDesc(1, 3, 4, 4);

  float avg_time = 0;
  modeB = 0;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(), &avg_time, modeB);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_per_act_fwd_channel3";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());

}