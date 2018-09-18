#include "test_pooling_forward.hpp"

TEST(pooling_fwd, func_check_zero_padding) {
  test_2dpool_desc_t pool(1, 1, 224, 224, 224 / 2, 224 / 2, 2, 2, 0, 0, 2, 2);
  Memory<float> srcData(pool.ih * pool.iw);
  Memory<float> dstDataCPU((224 / 2) * (224 / 2));
  Memory<float> dstDataGPU((224 / 2) * (224 / 2));

  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for(int i = 0; i < benchmark_iterations; i++){
      timer.restart();
      compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu());
      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1e6;
    }

    float* temp = dstDataGPU.getDataFromGPU();
    double avg_time = std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
    std::cout << "Average Time: " << avg_time << std::endl;

  std::string strt = "./result_unittest.csv";
  std::string testname = "func_check_pooling";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}
