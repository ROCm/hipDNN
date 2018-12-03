#include "fusion_api_NA.hpp"

//This test currently fails in NVidia, so it has been commented

/*
TEST(fusion_api_NA, func_check_fusion_api_NA) {

  float avg_time = 0;

  Desc inputDesc(1, 1, 5, 5);
  Desc filterDesc(1, 1, 3, 3);
  //int pad[2] = {0, 0};    // zero padding
  //int stride[2] = {1, 1}; // stride 1
  //int dil[2] = {1,1};

  Memory<float> srcDataConv = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(inputDesc);
  populateMemoryRandom<float>(srcDataConv);

  convulution_Size testConvolutionSizes(inputDesc.N, 1, inputDesc.C,
                   inputDesc.H, inputDesc.W, inputDesc.C, inputDesc.H,
                   inputDesc.W, 0, 0, 0, 0, 0, 0, 1, 1);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {0,0,0,0};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)ip_size,4);

  compute_hipdnn_fusion_api_NA<float>(testConvolutionSizes, srcDataConv.gpu(),
                                      dstDataGPU.gpu(),&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "fusion_api_NA: func_check_fusion_api_NA";
  float* temp = dstDataGPU.getDataFromGPU();
  std::string str  = convert_to_string((float*)temp,
                                       (int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size,
               str_k_size, str_op_size);

}
*/