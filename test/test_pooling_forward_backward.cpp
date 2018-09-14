#include "test_pooling_common.hpp"

TEST(pooling_fwd_back, func_check_fwd_bwd) {
  pool_fwd pool(1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  Memory<float> srcData(pool.ih * pool.iw);
  Memory<float> dstData(pool.oh * pool.ow);
  Memory<float> gradData(pool.ih * pool.iw);
  populateMemoryRandom<float>(srcData);
  
  high_resolution_timer_t timer;

   hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstData.gpu());

   int oheight = 4, owidth = 4;
  
   pool_bwd test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);
  
   hipdnn_pooling_backward(test_case, srcData.gpu(), gradData.gpu(), dstData.gpu());
  
    float* temp2 = gradData.getDataFromGPU();
    
    std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
    std::uint64_t timer_t = (time_elapsed / 1000.0);
    std::cout << "time taken: " << timer_t << " ms"<< std::endl;
    std::string strt = "./result_unittest.csv";
    std::string testname = "func_pooling_fwd_bwd";
    std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname, timer_t);
}
