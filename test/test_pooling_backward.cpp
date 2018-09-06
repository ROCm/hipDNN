#include "test_pooling_backward.hpp"

TEST(pooling_backward, func_check_pooling_stride_2x2) {
  int oheight = 2, owidth = 2;
  test_pooling_t test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);
  Memory<float> dataSrc(test_case.in * test_case.ichannel * test_case.iheight *
                        test_case.iwidth);
  Memory<float> dataGrad(test_case.in * test_case.ichannel * test_case.iheight *
                         test_case.iwidth);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  Memory<float> dataDst(test_case.in * test_case.ichannel * test_case.iheight *
                        test_case.iwidth);
  high_resolution_timer_t timer;
  compute_hipdnn_pooling_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                  dataDst.gpu());
    std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
    std::cout << "time taken: " << (time_elapsed / 1000.0) << " ms"<< std::endl;
    std::string strt = "./result_unittest.csv";
    std::string testname = "func_check_pooling_stride_2x2";
    float* temp = dataDst.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname);
}