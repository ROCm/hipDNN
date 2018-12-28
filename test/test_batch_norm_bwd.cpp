#include "test_batchnorm_common.hpp"

int mode; //0:SPATIAL 1:Per_Activation
int acc_grad;

TEST(BNorm_Backward, func_check_spatial_no_grad_bwd) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 1);

  float avg_time = 0;
  mode = 0;
  acc_grad = 0;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(inputDesc);
  Memory<float> resultBnScaleDiff = createMemory<float>(outputDesc);
  Memory<float> resultBnBiasDiff = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);
  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_bwd<float>(BN_sizes, srcData.gpu(), dstDataGPU.gpu(),
                                      resultBnScaleDiff.gpu(),
                                      resultBnBiasDiff.gpu(), &avg_time, mode, acc_grad);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname1 = "BNorm_Backward: func_check_spatial_no_grad_bwd_dx";
  std::string testname2 = "BNorm_Backward: func_check_spatial_no_grad_bwd_resultBnScaleDiff";
  std::string testname3 = "BNorm_Backward: func_check_spatial_no_grad_bwd_resultBnBiasDiff";
  std::string filename="BNorm_backward.csv";

  float* temp1 = dstDataGPU.getDataFromGPU();
  float* temp2 = resultBnScaleDiff.getDataFromGPU();
  float* temp3 = resultBnBiasDiff.getDataFromGPU();

  std::string str1  = convert_to_string2((float*)temp1,
                                    (int)dstDataGPU.get_num_elements());
  std::string str2  = convert_to_string2((float*)temp2,
                                    (int)resultBnScaleDiff.get_num_elements());
  std::string str3  = convert_to_string2((float*)temp3,
                                    (int)resultBnBiasDiff.get_num_elements());

  write_to_csv(strt, str1, testname1, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str2, testname2, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str3, testname3, avg_time, str_ip_size, str_k_size,
              str_op_size);

  dump_result_csv(filename, testname1, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname2, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultBnBiasDiff.get_num_elements());

}

TEST(BNorm_Backward, func_check_BNorm_bwd_per_act_mode_no_grad) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 6);

  float avg_time = 0;
  mode = 1;
  acc_grad = 0;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(inputDesc);
  Memory<float> resultBnScaleDiff = createMemory<float>(outputDesc);
  Memory<float> resultBnBiasDiff = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_bwd<float>(BN_sizes, srcData.gpu(), dstDataGPU.gpu(),
                                      resultBnScaleDiff.gpu(),
                                      resultBnBiasDiff.gpu(), &avg_time, mode, acc_grad);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname1 = "BNorm_Backward: per_act_mode_no_grad_bwd_dx";
  std::string testname2 = "BNorm_Backward: per_act_mode_no_grad_bwd_resultBnScaleDiff";
  std::string testname3 = "BNorm_Backward: per_act_mode_no_grad_bwd_resultBnBiasDiff";
  std::string filename="BNorm_backward.csv";

  float* temp1 = dstDataGPU.getDataFromGPU();
  float* temp2 = resultBnScaleDiff.getDataFromGPU();
  float* temp3 = resultBnBiasDiff.getDataFromGPU();

  std::string str1  = convert_to_string2((float*)temp1,
                                     (int)dstDataGPU.get_num_elements());
  std::string str2  = convert_to_string2((float*)temp2,
                                     (int)resultBnScaleDiff.get_num_elements());
  std::string str3  = convert_to_string2((float*)temp3,
                                     (int)resultBnBiasDiff.get_num_elements());

  write_to_csv(strt, str1, testname1, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str2, testname2, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str3, testname3, avg_time, str_ip_size, str_k_size,
              str_op_size);

  dump_result_csv(filename, testname1, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname2, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultBnBiasDiff.get_num_elements());

  }

TEST(BNorm_Backward, func_check_spatial_grad_bwd) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 1);

  float avg_time = 0;
  mode = 0;
  acc_grad = 1;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(inputDesc);
  Memory<float> resultBnScaleDiff = createMemory<float>(outputDesc);
  Memory<float> resultBnBiasDiff = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);
  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_bwd<float>(BN_sizes, srcData.gpu(), dstDataGPU.gpu(),
                                     resultBnScaleDiff.gpu(),
                                     resultBnBiasDiff.gpu(), &avg_time, mode, acc_grad);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname1 = "BNorm_Backward: func_check_spatial_grad_bwd_dx";
  std::string testname2 = "BNorm_Backward: func_check_spatial_grad_bwd_resultBnScaleDiff";
  std::string testname3 = "BNorm_Backward: func_check_spatial_grad_bwd_resultBnBiasDiff";
  std::string filename="BNorm_backward.csv";

  float* temp1 = dstDataGPU.getDataFromGPU();
  float* temp2 = resultBnScaleDiff.getDataFromGPU();
  float* temp3 = resultBnBiasDiff.getDataFromGPU();

  std::string str1  = convert_to_string2((float*)temp1,
                                     (int)dstDataGPU.get_num_elements());
  std::string str2  = convert_to_string2((float*)temp2,
                                     (int)resultBnScaleDiff.get_num_elements());
  std::string str3  = convert_to_string2((float*)temp3,
                                     (int)resultBnBiasDiff.get_num_elements());

  write_to_csv(strt, str1, testname1, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str2, testname2, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str3, testname3, avg_time, str_ip_size, str_k_size,
              str_op_size);

  dump_result_csv(filename, testname1, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname2, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultBnBiasDiff.get_num_elements());

  }

TEST(BNorm_Backward, func_check_BNorm_bwd_per_act_mode_grad) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 6);

  float avg_time = 0;
  mode = 1;
  acc_grad = 1;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(inputDesc);
  Memory<float> resultBnScaleDiff = createMemory<float>(outputDesc);
  Memory<float> resultBnBiasDiff = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);
  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_bwd<float>(BN_sizes, srcData.gpu(), dstDataGPU.gpu(),
                                      resultBnScaleDiff.gpu(),
                                      resultBnBiasDiff.gpu(), &avg_time, mode, acc_grad);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname1 = "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad_bwd_dx";
  std::string testname2 = "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad_bwd_resultBnScaleDiff";
  std::string testname3 = "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad_bwd_resultBnBiasDiff";
  std::string filename="BNorm_backward.csv";

  float* temp1 = dstDataGPU.getDataFromGPU();
  float* temp2 = resultBnScaleDiff.getDataFromGPU();
  float* temp3 = resultBnBiasDiff.getDataFromGPU();

  std::string str1  = convert_to_string2((float*)temp1,
                                     (int)dstDataGPU.get_num_elements());
  std::string str2  = convert_to_string2((float*)temp2,
                                     (int)resultBnScaleDiff.get_num_elements());
  std::string str3  = convert_to_string2((float*)temp3,
                                     (int)resultBnBiasDiff.get_num_elements());

  write_to_csv(strt, str1, testname1, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str2, testname2, avg_time, str_ip_size, str_k_size,
              str_op_size);
  write_to_csv(strt, str3, testname3, avg_time, str_ip_size, str_k_size,
              str_op_size);

  dump_result_csv(filename, testname1, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname2, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultBnBiasDiff.get_num_elements());

}