#include "test_activation_common.hpp"

hipdnnActivationMode_t mode_a;

TEST(activation_fwd_bwd, func_test_int_activation) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

  mode_a = HIPDNN_ACTIVATION_RELU;

  activation_params_t test_case1(1, 1, 4, 4);
  activation_params_t test_case2(1, 1, 4, 4);

  Memory<float> dataSrc(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);
  Memory<float> dataGrad(test_case2.n * test_case2.channels * test_case2.height
                          * test_case2.width);
  Memory<float> dataDst(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};
  int op_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case1, dataSrc.gpu(),dataDst.gpu(),
                                    mode_a, &avg_time1);
  compute_hipdnn_activation_backward(test_case2, dataSrc.gpu(), dataGrad.gpu(),
                                     dataDst.gpu(), mode_a, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = dataGrad.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_fwd_bwd:func_test_int_activation";
  std::string filename="activation_forward_backward.csv";
  std::string str  = convert_to_string((float*)temp2,
                                       (int)dataGrad.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);

  dump_result_csv(filename, testname, temp2, (int)dataGrad.get_num_elements());

}

TEST(activation_fwd_bwd, func_int_activation_batch32) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

  mode_a = HIPDNN_ACTIVATION_RELU;

  activation_params_t test_case1(32, 1, 4, 4);
  activation_params_t test_case2(32, 1, 4, 4);

  Memory<float> dataSrc(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);
  Memory<float> dataGrad(test_case2.n * test_case2.channels * test_case2.height
                          * test_case2.width);
  Memory<float> dataDst(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};
  int op_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case1, dataSrc.gpu(),dataDst.gpu(),
                                    mode_a, &avg_time1);

  compute_hipdnn_activation_backward(test_case2, dataSrc.gpu(), dataGrad.gpu(),
                                     dataDst.gpu(), mode_a, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = dataGrad.getDataFromGPU();
  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_fwd_bwd:func_int_activation_batch32";
  std::string filename="activation_forward_backward.csv";
  std::string str  = convert_to_string((float*)temp2,
                                         (int)dataGrad.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
                 str_op_size);
  dump_result_csv(filename, testname, temp2, (int)dataGrad.get_num_elements());

}

TEST(activation_fwd_bwd, func_int_activation_batch64) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

  mode_a = HIPDNN_ACTIVATION_RELU;

  activation_params_t test_case1(64, 1, 4, 4);
  activation_params_t test_case2(64, 1, 4, 4);

  Memory<float> dataSrc(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);
  Memory<float> dataGrad(test_case2.n * test_case2.channels * test_case2.height
                          * test_case2.width);
  Memory<float> dataDst(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};
  int op_size[4] =  {test_case1.n, test_case1.channels, test_case1.height,
                     test_case1.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case1, dataSrc.gpu(),dataDst.gpu(),
                                    mode_a, &avg_time1);

  compute_hipdnn_activation_backward(test_case2, dataSrc.gpu(), dataGrad.gpu(),
                                     dataDst.gpu(), mode_a, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = dataGrad.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_fwd_bwd:func_int_activation_batch64";
  std::string filename="activation_forward_backward.csv";
  std::string str  = convert_to_string((float*)temp2,
                                       (int)dataGrad.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp2, (int)dataGrad.get_num_elements());
}

TEST(activation_fwd_bwd, func_int_activation_batch128) {

  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;

  mode_a = HIPDNN_ACTIVATION_RELU;

  activation_params_t test_case1(128, 1, 25, 25);
  activation_params_t test_case2(128, 1, 25, 25);

  Memory<float> dataSrc(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);
  Memory<float> dataGrad(test_case2.n * test_case2.channels * test_case2.height
                          * test_case2.width);
  Memory<float> dataDst(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);

  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};
  int op_size[4] = {test_case1.n, test_case1.channels, test_case1.height,
                    test_case1.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case1, dataSrc.gpu(),dataDst.gpu(),
                                    mode_a, &avg_time1);

  compute_hipdnn_activation_backward(test_case2, dataSrc.gpu(), dataGrad.gpu(),
                                     dataDst.gpu(), mode_a, &avg_time2);

  avg_time = (avg_time1 + avg_time2);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp2 = dataGrad.getDataFromGPU();

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_fwd_bwd:func_int_activation_batch128";
  std::string filename="activation_forward_backward.csv";
  std::string str  = convert_to_string((float*)temp2,
                                       (int)dataGrad.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp2, (int)dataGrad.get_num_elements());

}