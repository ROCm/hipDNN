#include "test_pooling_common.hpp"

TEST(pooling_fwd_back, func_check_fwd_bwd) {
  int oheight = 4, owidth = 4;
  pool_fwd pool(1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  pool_bwd test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);
  Memory<float> srcData(pool.ih * pool.iw);
  Memory<float> dstData(pool.oh * pool.ow);
  Memory<float> gradData(pool.ih * pool.iw);
  populateMemoryRandom<float>(srcData);
       
  high_resolution_timer_t timer;

    std::vector<double> time_vector(benchmark_iterations);
    for(int i = 0; i < benchmark_iterations; i++){
      timer.restart();
      hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstData.gpu());
      hipdnn_pooling_backward(test_case, srcData.gpu(), gradData.gpu(), dstData.gpu());
      hipDeviceSynchronize();
      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1000.0;
    }
    double avg_time = std::accumulate(time_vector.begin() + 10, time_vector.end(), 0.0) / (benchmark_iterations - 10);
    std::cout << "Average Time: " << avg_time << std::endl;

    float* temp2 = gradData.getDataFromGPU();
   
    std::string strt = "./result_unittest.csv";
    std::string testname = "func_pooling_fwd_bwd";
    std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname, avg_time);
}
