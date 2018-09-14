#include "test_activation_backward.hpp"
#include "test_activation_forward.hpp"

TEST(activation_fwd_bwd, func_test_fwd_bwd_activation) {
  
  activation_fwd_params test_case1(1, 1, 4, 4);
  activation_params_t test_case2(1, 1, 4, 4);

  Memory<float> dataSrc(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);
  Memory<float> dataGrad(test_case2.n * test_case2.channels * test_case2.height 
                          * test_case2.width);
  Memory<float> dataDst(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);

  populateMemoryRandom(dataSrc);
  
  high_resolution_timer_t timer;

  compute_hipdnn_activation_forward(test_case1, dataSrc.gpu(),dataDst.gpu());
  compute_hipdnn_activation_backward(test_case2, dataSrc.gpu(), dataGrad.gpu(),
                                     dataDst.gpu());

    float* temp2 = dataGrad.getDataFromGPU();

    std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
    std::uint64_t timer_t = (time_elapsed / 1000.0);
    std::cout << "time taken: " << timer_t << " ms"<< std::endl;
    std::string strt = "./result_unittest.csv";
    std::string testname = "func_test_fwd_bwd_activation";
    std::string str  = convert_to_string((float*)temp2,(int)dataGrad.get_num_elements());
    write_to_csv(strt, str, testname,timer_t);                                      
}
