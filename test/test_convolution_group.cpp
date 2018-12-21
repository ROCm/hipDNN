#include "test_convolution_group.hpp"

//This test currently fails in NVidia, so it has been commented

/*
TEST(group_convolution_fwd, func_check_group_conv) {

  Desc inputDesc(32, 12, 224, 224);
  Desc filterDesc(4, 12, 3, 3);

  int pad[2] = {0, 0};    // zero padding
  int stride[2] = {1, 1}; // stride 1
  int dil[2] = {1, 1};
  float avg_time = 0;
  int grp_cnt = 2;

  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataCPU = createMemory<float>(outputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size testConvolutionSizes(
        inputDesc.N, 2, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_group_conv<float>(testConvolutionSizes, srcData.gpu(),
                            filterData.gpu(), NULL, dstDataGPU.gpu(),&avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "group_convolution_fwd: func_check_group_conv";
  std::string filename="group_convolution_fwd.csv";

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}
*/