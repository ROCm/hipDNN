#include "test_pooling_backward.hpp"

TEST(pooling_backward, func_check_pooling_stride_2x2) {

  int oheight = 4, owidth = 4;
  test_pooling_t test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);
  Memory<float> dataSrc(16);
  Memory<float> dataGrad(16);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  Memory<float> dataDst(test_case.on * test_case.ochannel * test_case.oheight *
                        test_case.owidth);

  size_t ip_size[4] = {test_case.in, test_case.ichannel, test_case.iheight, test_case.iwidth};
  size_t k_size[4] = {test_case.in, test_case.ichannel, test_case.wheight, test_case.wwidth};
  int op_size[4] =  {test_case.on, test_case.ochannel, test_case.oheight, test_case.owidth};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for(int i = 0; i < benchmark_iterations; i++){
      timer.restart();
  compute_hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                  dataDst.gpu());
      std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
      time_vector[i] = (double)time_elapsed / 1e6;
    }
    double avg_time = std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);
    std::cout << "Average Time: " << avg_time << std::endl;

    std::string strt = "./result_unittest.csv";
    std::string testname = "func_check_pooling_stride_2x2";
    float* temp = dataGrad.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
}
