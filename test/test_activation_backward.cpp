#include "test_activation_backward.hpp"

TEST(activation_backward, func_test_naive_activation) {
  activation_params_t test_case(1, 1, 224, 244);
  Memory<float> dataSrc(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  Memory<float> dataGrad(test_case.n * test_case.channels * test_case.height *
                         test_case.width);
  Memory<float> dataDst(test_case.n * test_case.channels * test_case.height *
                        test_case.width);
  populateMemoryRandom(dataSrc);
  populateMemoryRandom(dataGrad);
  compute_hipdnn_activation_backward(test_case, dataSrc.gpu(), dataGrad.gpu(),
                                     dataDst.gpu());
    std::string strt = "./result_unittest.csv";
    std::string testname = "func_test_naive_activation";
    float* temp = dataDst.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dataDst.get_num_elements());
    write_to_csv(strt, str, testname);                                     
}