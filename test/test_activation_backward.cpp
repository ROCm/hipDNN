#include "test_activation_common.hpp"

hipdnnActivationMode_t actB_mode;

TEST(activation_backward, func_test_activation_RELU) {

  activation_params_t test_case(1, 1, 224, 244);
  float avg_time = 0;
  alpha = 1.f;
  beta = 0.f;
  actB_mode = HIPDNN_ACTIVATION_RELU;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    actB_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<float>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            actB_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_backward:func_test_activation_RELU";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());

}

TEST(activation_backward, func_test_activation_SIGMOID) {

  activation_params_t test_case(1, 1, 224, 244);
  float avg_time = 0;
  alpha = 1.f;
  beta = 0.5f;
  actB_mode = HIPDNN_ACTIVATION_SIGMOID;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    actB_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<float>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            actB_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_backward:func_test_activation_SIGMOID";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());

}

TEST(activation_backward, func_test_activation_TANH) {

  activation_params_t test_case(1, 1, 224, 244);
  float avg_time = 0;
  alpha = 2.f;
  beta = 0.f;
  actB_mode = HIPDNN_ACTIVATION_TANH;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    actB_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<float>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            actB_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_backward:func_test_activation_TANH";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());

}

TEST(activation_backward, func_test_activation_batch32) {

  activation_params_t test_case(32, 1, 224, 244);
  float avg_time = 0;
  alpha = 2.f;
  beta = 1.f;
  actB_mode = HIPDNN_ACTIVATION_RELU;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] =  {test_case.n, test_case.channels, test_case.height,
                     test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    actB_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<float>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            actB_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_backward:func_test_activation_batch32";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size,
               str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());

}

TEST(activation_backward, func_test_activation_batch64) {

  activation_params_t test_case(64, 1, 224, 244);
  float avg_time = 0;
  alpha = 0.5f;
  beta = 0.f;
  actB_mode = HIPDNN_ACTIVATION_RELU;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);

  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] =  {test_case.n, test_case.channels, test_case.height,
                     test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    actB_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<float>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            actB_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_backward:func_test_activation_batch64";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size,
               str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());

}

TEST(activation_backward, func_test_activation_batch128) {

  activation_params_t test_case(128, 1, 4, 4);
  float avg_time = 0;
  alpha = 2.f;
  beta = 0.5f;
  actB_mode = HIPDNN_ACTIVATION_RELU;

  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};
  int op_size[4] = {test_case.n, test_case.channels, test_case.height,
                    test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(),
                                    actB_mode, alpha, beta, &avg_time);

  compute_hipdnn_activation_backward<float>(test_case, dataSrc.gpu(),
                                            dataGrad.gpu(), dataDst.gpu(),
                                            actB_mode, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "activation_backward:func_test_activation_batch128";
  std::string filename="activation_backward.csv";

  float* temp = dataDst.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataDst.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size,
               str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataDst.get_num_elements());
}