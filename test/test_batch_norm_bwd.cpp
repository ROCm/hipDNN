#include "test_batch_norm_bwd.hpp"

TEST(BNorm_Backward, func_check_spatial_no_grad_bwd) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 1);

  float avg_time = 0;
  int flag = 0;

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
                                      resultBnBiasDiff.gpu(), &avg_time, flag);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Backward: func_check_spatial_no_grad_bwd";
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

  std::ostringstream os;
  os <<  "\" dx: " << str1 << ", resultBnScaleDiff: " << str2
     << ", resultBnBiasDiff: " << str3 << "\"";

  std::string str(os.str());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
              str_op_size);
  dump_result_csv(filename, testname, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname, temp3, (int)resultBnBiasDiff.get_num_elements());

}

TEST(BNorm_Backward, func_check_BNorm_bwd_per_act_mode_no_grad) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 6);

  float avg_time = 0;
  int flag = 1;

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
                                      resultBnBiasDiff.gpu(), &avg_time, flag);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_no_grad";
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

  std::ostringstream os;
  os <<  "\" dx: " << str1 << ", resultBnScaleDiff: " << str2
     << ", resultBnBiasDiff: " << str3 << "\"";

  std::string str(os.str());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname, temp3, (int)resultBnBiasDiff.get_num_elements());

  }

TEST(BNorm_Backward, func_check_spatial_grad_bwd) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 1);

  float avg_time = 0;
  int flag = 2;

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
                                     resultBnBiasDiff.gpu(), &avg_time, flag);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Backward: func_check_spatial_grad_bwd";
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

    std::ostringstream os;
    os <<  "\" dx: " << str1 << ", resultBnScaleDiff: " << str2
       << ", resultBnBiasDiff: " << str3 << "\"";

    std::string str(os.str());

    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
                 str_op_size);

  dump_result_csv(filename, testname, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname, temp3, (int)resultBnBiasDiff.get_num_elements());

  }

TEST(BNorm_Backward, func_check_BNorm_bwd_per_act_mode_grad) {

  Desc inputDesc(6, 1, 1, 6);
  Desc outputDesc(1, 1, 1, 6);

  float avg_time = 0;
  int flag = 3;

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
                                      resultBnBiasDiff.gpu(), &avg_time, flag);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Backward: func_check_BNorm_bwd_per_act_mode_grad";
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

  std::ostringstream os;
  os <<  "\" dx: " << str1 << ", resultBnScaleDiff: " << str2
     << ", resultBnBiasDiff: " << str3 << "\"";

  std::string str(os.str());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp1, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname, temp2, (int)resultBnScaleDiff.get_num_elements());
  dump_result_csv(filename, testname, temp3, (int)resultBnBiasDiff.get_num_elements());

}