#include "test_batchnorm_fwd.hpp"

TEST(BNorm_Fwd, func_check_naive_BNorm_forward) {
  Desc inputDesc(3, 2, 5, 5);
    Desc filterDesc(1, 2, 3, 3);
    int pad[2] = {0,0};    // padding 2
    int stride[2] = {1,1}; // stride 2
    float avg_time = 0;
    int dil[2] = {1,1};

    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride,dil);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    convulution_Size convulution_sizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], 1, 1);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_batchnorm_fwd<float>(convulution_sizes, srcData.gpu(),
                                filterData.gpu(), dstDataGPU.gpu(), &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "BNorm_Fwd: func_check_naive_BNorm_forward";
    float* temp = dstDataGPU.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  }
