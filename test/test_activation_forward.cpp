#include "test_activation_forward.hpp"

TEST(activation_forward, func_test_fwd_activation) {
  float avg_time = 0;
  activation_fwd_params test_case(1, 1, 4, 4);
  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height, test_case.width};
  int op_size[4] =  {test_case.n, test_case.channels, test_case.height, test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(), &avg_time);
  
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dataDst.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "activation_forward:func_test_fwd_activation";
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(activation_forward, func_fwd_activation_batch32) {
  float avg_time = 0;
  activation_fwd_params test_case(32, 1, 4, 4);
  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height, test_case.width};
  int op_size[4] =  {test_case.n, test_case.channels, test_case.height, test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(), &avg_time);
  
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dataDst.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "activation_forward:func_fwd_activation_batch32";
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(activation_forward, func_fwd_activation_batch64) {
  float avg_time = 0;
  activation_fwd_params test_case(64, 1, 4, 4);
  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  populateMemoryRandom(dataSrc);

  int ip_size[4] = {test_case.n, test_case.channels, test_case.height, test_case.width};
  int op_size[4] =  {test_case.n, test_case.channels, test_case.height, test_case.width};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = "NIL";
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu(), &avg_time);
  
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  float* temp = dataDst.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "activation_forward:func_fwd_activation_batch64";
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}
