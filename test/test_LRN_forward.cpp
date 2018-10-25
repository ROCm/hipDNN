#include "test_LRN_forward.hpp"

TEST(LRN_fwd, func_check_naive_LRN) {
  Desc inputDesc(1, 3, 7, 7);
    Desc filterDesc(1, 3, 3, 3);
    int pad[2] = {2, 2};    // padding 2
    int stride[2] = {2, 2}; // stride 2
    float avg_time = 0;
    int dil[2] = {1,1};

    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), dstDataGPU.gpu(), &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "LRN_fwd:func_check_naive_LRN";
    float* temp = dstDataGPU.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  }

TEST(LRN_fwd, func_check_LRN_batch16) {
  Desc inputDesc(16, 3, 60, 60);
    Desc filterDesc(21, 3, 30, 30);
    int pad[2] = {0, 0};    // padding 2
    int stride[2] = {2, 2}; // stride 2
    float avg_time = 0;
    int dil[2] = {1,1};

    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), dstDataGPU.gpu(), &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "LRN_fwd:func_check_LRN_batch16";
    float* temp = dstDataGPU.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  }

TEST(LRN_fwd, func_check_LRN_batch32) {
 Desc inputDesc(16, 3, 60, 60);
    Desc filterDesc(21, 3, 30, 30);
    int pad[2] = {0, 0};    // padding 2
    int stride[2] = {2, 2}; // stride 2
    float avg_time = 0;
    int dil[2] = {1,1};

    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), dstDataGPU.gpu(), &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "LRN_fwd:func_check_LRN_batch32";
    float* temp = dstDataGPU.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  }

TEST(LRN_fwd, func_check_LRN_batch64) {
  Desc inputDesc(64, 3, 7, 7);
    Desc filterDesc(21, 3, 3, 3);
    int pad[2] = {2, 2};    // padding 2
    int stride[2] = {2, 2}; // stride 2
    float avg_time = 0;
    int dil[2] = {1,1};

    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), dstDataGPU.gpu(), &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "LRN_fwd:func_check_LRN_batch64";
    float* temp = dstDataGPU.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  }

TEST(LRN_fwd, func_check_LRN_batch128) {
  Desc inputDesc(128, 3, 7, 7);
    Desc filterDesc(21, 3, 3, 3);
    int pad[2] = {2, 2};    // padding 2
    int stride[2] = {2, 2}; // stride 2
    float avg_time = 0;
    int dil[2] = {1,1};

    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  compute_hipdnn_LRN_fwd<float>(testConvolutionSizes, srcData.gpu(),
                                filterData.gpu(), dstDataGPU.gpu(), &avg_time);

   std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "LRN_fwd:func_check_LRN_batch128";
    float* temp = dstDataGPU.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  }
