#include "test_activation_forward.hpp"

TEST(activation_forward, func_test_fwd_activation) {

  activation_fwd_params test_case(1, 1, 4, 4);
  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  populateMemoryRandom(dataSrc);

  high_resolution_timer_t timer;
 
  std::vector<double> time_vector(benchmark_iterations);
    for(int i = 0; i < benchmark_iterations; i++){
      timer.restart();
      compute_hipdnn_activation_forward(test_case, dataSrc.gpu(), dataDst.gpu());
      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      hipDeviceSynchronize();
      time_vector[i] = (double)time_elapsed / 1000.0;
    }
    double avg_time = std::accumulate(time_vector.begin() + 10, time_vector.end(), 0.0) / (benchmark_iterations - 10);
    std::cout << "Average Time: " << avg_time << std::endl;

  float* temp = dataDst.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "func_test_fwd_activation";
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname,avg_time);                                     
}
