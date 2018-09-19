#include "test_convolution_backward_filter.hpp"

TEST(convolution_bwd_filter, func_check_backward_conv_filter) {
    Desc inputDesc(1, 3, 224, 224);
    Desc filterDesc(21, 3, 3, 3);
    int pad[2] = {0, 0};    // zero padding
    int stride[2] = {1, 1}; // stride 1
    Desc outputDesc =
        calculateConv2DOutputDesc_bwd(inputDesc, filterDesc, pad, stride);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> gradData = createMemory<float>(filterDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);

    test_convolution_bwd_filter testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], 1, 1);

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

high_resolution_timer_t timer;
std::vector<double> time_vector(benchmark_iterations, 0);
    for(int i = 0; i < benchmark_iterations; i++){
      timer.restart();
        compute_hipdnn_conv_bwd_filter<float>(testConvolutionSizes, srcData.gpu(),
                                 filterData.gpu(), gradData.gpu(), NULL, dstDataGPU.gpu());
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1e6;
    }
    double avg_time = std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
    std::cout << "Average Time: " << avg_time << std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "convolution_bwd_filter:func_check_backward_conv_filter";
    float* temp = gradData.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
 }

