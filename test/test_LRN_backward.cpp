#include "test_LRN_backward.hpp"

TEST(LRN_bwd, func_check_naive_LRN_backward) {

  Desc inputDesc(1, 3, 7, 7);
  float avg_time = 0;

  Desc gradDesc(1, 3, 7, 7);
  Desc outputDesc(1, 3, 7, 7);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> gradDataGPU = createMemory<float>(gradDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);

  populateMemoryRandom<float>(srcData);

  LRN_params_t LRN_params(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_backward<float>(LRN_params, srcData.gpu(),
                                gradDataGPU.gpu(), dstDataGPU.gpu(), &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "LRN_bwd: func_check_naive_LRN_backward";
  std::string filename="LRN_bwd.csv";

  float* temp = gradDataGPU.getDataFromGPU();
  std::string str  = convert_to_string((float*)temp,
                                       (int)gradDataGPU.get_num_elements());

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)gradDataGPU.get_num_elements());
}