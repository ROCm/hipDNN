#include "test_pooling_common.hpp"

TEST(pooling_fwd_back, func_check_fwd_bwd) {
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;
  int oheight = 4, owidth = 4;
  test_pooling_descriptor pool(1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  pool_bwd test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);
  Memory<float> srcData(pool.ih * pool.iw);
  Memory<float> dstData(pool.oh * pool.ow);
  Memory<float> gradData(pool.ih * pool.iw);
  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.ih, pool.iw};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstData.gpu(), &avg_time1);
  hipdnn_pooling_backward<float>(test_case, srcData.gpu(), gradData.gpu(), dstData.gpu(), &avg_time2);
     
    avg_time = (avg_time1 + avg_time2);

    std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    float* temp2 = gradData.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_intg:func_pooling_fwd_bwd";
    std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_fwd_back, func_int_batch32) {
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;
  int oheight = 4, owidth = 4;
  test_pooling_descriptor pool(32, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  pool_bwd test_case(32, 1, 4, 4, 2, 2, 0, 0, 2, 2, 32, 1, oheight, owidth);
  Memory<float> srcData(pool.mb * pool.c * pool.ih * pool.iw);
  Memory<float> dstData(pool.mb * pool.c * pool.oh * pool.ow);
  Memory<float> gradData(pool.mb * pool.c * pool.ih * pool.iw);
  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.ih, pool.iw};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstData.gpu(), &avg_time1);
  hipdnn_pooling_backward<float>(test_case, srcData.gpu(), gradData.gpu(), dstData.gpu(), &avg_time2);
     
    avg_time = (avg_time1 + avg_time2);

    std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    float* temp2 = gradData.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_intg:func_int_batch32";
    std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_fwd_back, func_int_batch64) {
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;
  int oheight = 4, owidth = 4;
  test_pooling_descriptor pool(64, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  pool_bwd test_case(64, 1, 4, 4, 2, 2, 0, 0, 2, 2, 64, 1, oheight, owidth);
  Memory<float> srcData(pool.mb * pool.c * pool.ih * pool.iw);
  Memory<float> dstData(pool.mb * pool.c * pool.oh * pool.ow);
  Memory<float> gradData(pool.mb * pool.c * pool.ih * pool.iw);
  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.ih, pool.iw};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstData.gpu(), &avg_time1);
  hipdnn_pooling_backward<float>(test_case, srcData.gpu(), gradData.gpu(), dstData.gpu(), &avg_time2);
     
    avg_time = (avg_time1 + avg_time2);

    std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    float* temp2 = gradData.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_intg:func_int_batch64";
    std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}

TEST(pooling_fwd_back, func_int_batch128) {
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0;
  int oheight = 4, owidth = 4;
  test_pooling_descriptor pool(128, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  pool_bwd test_case(128, 1, 4, 4, 2, 2, 0, 0, 2, 2, 128, 1, oheight, owidth);
  Memory<float> srcData(pool.mb * pool.c * pool.ih * pool.iw);
  Memory<float> dstData(pool.mb * pool.c * pool.oh * pool.ow);
  Memory<float> gradData(pool.mb * pool.c * pool.ih * pool.iw);
  populateMemoryRandom<float>(srcData);

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.ih, pool.iw};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  hipdnn_pooling_forward<float>(pool, srcData.gpu(), dstData.gpu(), &avg_time1);
  hipdnn_pooling_backward<float>(test_case, srcData.gpu(), gradData.gpu(), dstData.gpu(), &avg_time2);
     
    avg_time = (avg_time1 + avg_time2);

    std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

    float* temp2 = gradData.getDataFromGPU();

    std::string strt = "./result_unittest.csv";
    std::string testname = "pooling_intg:func_int_batch128";
    std::string str  = convert_to_string((float*)temp2,(int)gradData.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
}
