#include "test_batchnorm_common.hpp"

TEST(BNorm_Fwd_train, func_check_spatial_fwd) {
  Desc inputDesc(1, 1, 4, 4);
  Desc outputDesc(1, 1, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 1, 1, 1);

  float avg_time = 0;
  bn_mode = HIPDNN_BATCHNORM_SPATIAL;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> resultRunningMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultRunningVariance = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveVariance = createMemory<float>(bnScaleBiasMeanVarDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(resultRunningMean);
  populateMemoryRandom<float>(resultRunningVariance);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(),
                                            resultRunningMean.gpu(),
                                            resultRunningVariance.gpu(),
                                            resultSaveMean.gpu(),
                                            resultSaveVariance.gpu(), dataType,
                                            &avg_time, bn_mode);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_spatial_fwd";
  std::string testname1 = "BNorm_Fwd_train: func_check_spatial_fwd_resultRunningMean";
  std::string testname2 = "BNorm_Fwd_train: func_check_spatial_fwd_resultRunningVariance";
  std::string testname3 = "BNorm_Fwd_train: func_check_spatial_fwd_resultSaveMean";
  std::string testname4 = "BNorm_Fwd_train: func_check_spatial_fwd_resultSaveVariance";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();
  float* temp1 = resultRunningMean.getDataFromGPU();
  float* temp2 = resultRunningVariance.getDataFromGPU();
  float* temp3 = resultSaveMean.getDataFromGPU();
  float* temp4 = resultSaveVariance.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());
  std::string str1  = convert_to_string((float*)temp1,
                                       (int)resultRunningMean.get_num_elements());
  std::string str2  = convert_to_string((float*)temp2,
                                       (int)resultRunningVariance.get_num_elements());
  std::string str3  = convert_to_string((float*)temp3,
                                       (int)resultSaveMean.get_num_elements());
  std::string str4  = convert_to_string((float*)temp4,
                                       (int)resultSaveVariance.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str1, testname1,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str2, testname2,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str3, testname3,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str4, testname4,avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname1, temp1, (int)resultRunningMean.get_num_elements());
  dump_result_csv(filename, testname2, temp2, (int)resultRunningVariance.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultSaveMean.get_num_elements());
  dump_result_csv(filename, testname4, temp4, (int)resultSaveVariance.get_num_elements());

}

TEST(BNorm_Fwd_train, func_check_per_act_fwd) {
  Desc inputDesc(1, 1, 4, 4);
  Desc outputDesc(1, 1, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 1, 4, 4);

  float avg_time = 0;
  bn_mode = HIPDNN_BATCHNORM_PER_ACTIVATION;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> resultRunningMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultRunningVariance = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveVariance = createMemory<float>(bnScaleBiasMeanVarDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(resultRunningMean);
  populateMemoryRandom<float>(resultRunningVariance);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(),
                                            resultRunningMean.gpu(),
                                            resultRunningVariance.gpu(),
                                            resultSaveMean.gpu(),
                                            resultSaveVariance.gpu(), dataType,
                                            &avg_time, bn_mode);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_per_act_fwd";
  std::string testname1 = "BNorm_Fwd_train: func_check_per_act_fwd_resultRunningMean";
  std::string testname2 = "BNorm_Fwd_train: func_check_per_act_fwd_resultRunningVariance";
  std::string testname3 = "BNorm_Fwd_train: func_check_per_act_fwd_resultSaveMean";
  std::string testname4 = "BNorm_Fwd_train: func_check_per_act_fwd_resultSaveVariance";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();
  float* temp1 = resultRunningMean.getDataFromGPU();
  float* temp2 = resultRunningVariance.getDataFromGPU();
  float* temp3 = resultSaveMean.getDataFromGPU();
  float* temp4 = resultSaveVariance.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());
  std::string str1  = convert_to_string((float*)temp1,
                                       (int)resultRunningMean.get_num_elements());
  std::string str2  = convert_to_string((float*)temp2,
                                       (int)resultRunningVariance.get_num_elements());
  std::string str3  = convert_to_string((float*)temp3,
                                       (int)resultSaveMean.get_num_elements());
  std::string str4  = convert_to_string((float*)temp4,
                                       (int)resultSaveVariance.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str1, testname1,avg_time, str_ip_size, str_k_size,
               str_op_size);
  //write_to_csv(strt, str2, testname2,avg_time, str_ip_size, str_k_size,
  //             str_op_size);
  write_to_csv(strt, str3, testname3,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str4, testname4,avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname1, temp1, (int)resultRunningMean.get_num_elements());
  //dump_result_csv(filename, testname2, temp2, (int)resultRunningVariance.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultSaveMean.get_num_elements());
  dump_result_csv(filename, testname4, temp4, (int)resultSaveVariance.get_num_elements());

}

TEST(BNorm_Fwd_train, func_check_spatial_fwd_channel3) {
  Desc inputDesc(1, 3, 4, 4);
  Desc outputDesc(1, 3, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 3, 1, 1);

  float avg_time = 0;
  bn_mode = HIPDNN_BATCHNORM_SPATIAL;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> resultRunningMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultRunningVariance = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveVariance = createMemory<float>(bnScaleBiasMeanVarDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(resultRunningMean);
  populateMemoryRandom<float>(resultRunningVariance);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(),
                                            resultRunningMean.gpu(),
                                            resultRunningVariance.gpu(),
                                            resultSaveMean.gpu(),
                                            resultSaveVariance.gpu(), dataType,
                                            &avg_time, bn_mode);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_spatial_fwd_channel3";
  std::string testname1 = "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultRunningMean";
  std::string testname2 = "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultRunningVariance";
  std::string testname3 = "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultSaveMean";
  std::string testname4 = "BNorm_Fwd_train: func_check_spatial_fwd_channel3_resultSaveVariance";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();
  float* temp1 = resultRunningMean.getDataFromGPU();
  float* temp2 = resultRunningVariance.getDataFromGPU();
  float* temp3 = resultSaveMean.getDataFromGPU();
  float* temp4 = resultSaveVariance.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());
  std::string str1  = convert_to_string((float*)temp1,
                                       (int)resultRunningMean.get_num_elements());
  std::string str2  = convert_to_string((float*)temp2,
                                       (int)resultRunningVariance.get_num_elements());
  std::string str3  = convert_to_string((float*)temp3,
                                       (int)resultSaveMean.get_num_elements());
  std::string str4  = convert_to_string((float*)temp4,
                                       (int)resultSaveVariance.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str1, testname1,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str2, testname2,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str3, testname3,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str4, testname4,avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname1, temp1, (int)resultRunningMean.get_num_elements());
  dump_result_csv(filename, testname2, temp2, (int)resultRunningVariance.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultSaveMean.get_num_elements());
  dump_result_csv(filename, testname4, temp4, (int)resultSaveVariance.get_num_elements());

}

TEST(BNorm_Fwd_train, func_check_per_act_fwd_channel3) {
  Desc inputDesc(1, 3, 4, 4);
  Desc outputDesc(1, 3, 4, 4);
  Desc bnScaleBiasMeanVarDesc(1, 3, 4, 4);

  float avg_time = 0;
  bn_mode = HIPDNN_BATCHNORM_PER_ACTIVATION;

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> resultRunningMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultRunningVariance = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveMean = createMemory<float>(bnScaleBiasMeanVarDesc);
  Memory<float> resultSaveVariance = createMemory<float>(bnScaleBiasMeanVarDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(resultRunningMean);
  populateMemoryRandom<float>(resultRunningVariance);

  BNorm_params_t BN_sizes(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd_train<float>(BN_sizes, srcData.gpu(),
                                            dstDataGPU.gpu(),
                                            resultRunningMean.gpu(),
                                            resultRunningVariance.gpu(),
                                            resultSaveMean.gpu(),
                                            resultSaveVariance.gpu(), dataType,
                                            &avg_time, bn_mode);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "BNorm_Fwd_train: func_check_per_act_fwd_channel3";
  std::string testname1 = "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultRunningMean";
  std::string testname2 = "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultRunningVariance";
  std::string testname3 = "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultSaveMean";
  std::string testname4 = "BNorm_Fwd_train: func_check_per_act_fwd_channel3_resultSaveVariance";
  std::string filename="BNorm_Fwd_train.csv";

  float* temp = dstDataGPU.getDataFromGPU();
  float* temp1 = resultRunningMean.getDataFromGPU();
  float* temp2 = resultRunningVariance.getDataFromGPU();
  float* temp3 = resultSaveMean.getDataFromGPU();
  float* temp4 = resultSaveVariance.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());
  std::string str1  = convert_to_string((float*)temp1,
                                       (int)resultRunningMean.get_num_elements());
  std::string str2  = convert_to_string((float*)temp2,
                                       (int)resultRunningVariance.get_num_elements());
  std::string str3  = convert_to_string((float*)temp3,
                                       (int)resultSaveMean.get_num_elements());
  std::string str4  = convert_to_string((float*)temp4,
                                       (int)resultSaveVariance.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str1, testname1,avg_time, str_ip_size, str_k_size,
               str_op_size);
  //write_to_csv(strt, str2, testname2,avg_time, str_ip_size, str_k_size,
  //             str_op_size);
  write_to_csv(strt, str3, testname3,avg_time, str_ip_size, str_k_size,
               str_op_size);
  write_to_csv(strt, str4, testname4,avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
  dump_result_csv(filename, testname1, temp1, (int)resultRunningMean.get_num_elements());
  //dump_result_csv(filename, testname2, temp2, (int)resultRunningVariance.get_num_elements());
  dump_result_csv(filename, testname3, temp3, (int)resultSaveMean.get_num_elements());
  dump_result_csv(filename, testname4, temp4, (int)resultSaveVariance.get_num_elements());

}