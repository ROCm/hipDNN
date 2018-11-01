#include "test_convolution_pooling_act_fwd_bwd_int.hpp"
#include "test_convolution_pooling_int.hpp"

TEST(convolution_pooling_act_fwd_bwd_intg, func_check_naive_conv_pool_act_fwd_bwd) {
  float avg_time = 0, avg_time1 = 0, avg_time2 = 0, avg_time3 = 0, avg_time4 = 0, avg_time5 = 0, avg_time6 = 0;
  int oheight = 4, owidth = 4;
  test_pooling_descriptor pool(1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);
  pool_bwd test_case(1, 1, 4, 4, 2, 2, 0, 0, 2, 2, 1, 1, oheight, owidth);

  Memory<float> dstData(pool.mb * pool.c * pool.oh * pool.ow);
  
  
  activation_params_t test_case1(1, 1, 4, 4);
 
  Memory<float> dataSrc_act(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);
  Memory<float> dataGrad_act(test_case1.n * test_case1.channels * test_case1.height
                          * test_case1.width);
  Memory<float> dataDst_act(test_case1.n * test_case1.channels * test_case1.height *
                        test_case1.width);

  populateMemoryRandom(dataSrc_act);

    Desc inputDesc(1, 1, 16, 16);
    Desc filterDesc(1, 1, 4, 4);
    int pad[2] = {0, 0};    // zero padding
    int stride[2] = {4, 4}; // stride 1
    int dil[2] = {1,1};
    Desc outputDesc =
        calculate_Dims(inputDesc, filterDesc, pad, stride, dil);
    Memory<float> srcDataConv = createMemory<float>(inputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    Memory<float> gradData1 = createMemory<float>(outputDesc);
    Memory<float> gradData2 = createMemory<float>(filterDesc);

    populateMemoryRandom<float>(srcDataConv);
    populateMemoryRandom<float>(filterData);
    convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

    convulution_Size testConvolutionSizes2(
        test_case1.n, 1, test_case1.channels, test_case1.height, test_case1.width, test_case1.channels, test_case1.height, test_case1.width, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  int ip_size_cf[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size_cf[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size_cf[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  int ip_size_pf[4] = {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};
  int k_size_pf[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size_pf[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  int ip_size_pb[4] = {pool.mb, pool.c, pool.oh, pool.ow};
  int k_size_pb[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size_pb[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  int ip_size_cb[4] = {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};
  int k_size_cb[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size_cb[4] =  {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};

  int ip_size_af[4] = {test_case1.n, test_case1.channels, test_case1.height, test_case1.width};
  int k_size_af[4] = {0,0,0,0};
  int op_size_af[4] =  {test_case1.n, test_case1.channels, test_case1.height, test_case1.width};

  int ip_size_ab[4] = {test_case1.n, test_case1.channels, test_case1.height, test_case1.width};
  int k_size_ab[4] = {0,0,0,0};
  int op_size_ab[4] =  {test_case1.n, test_case1.channels, test_case1.height, test_case1.width};

 std::string str_ip_size = integration_dims_to_string3(ip_size_cf,ip_size_pf,ip_size_af,ip_size_cb,ip_size_pb,ip_size_ab,"Conv_fwd","MP_fwd","act_fwd","Conv_bwd","MP_bwd", "act_bwd");
 std::string str_k_size = integration_dims_to_string3(k_size_cf,k_size_pf,k_size_af, k_size_cb,k_size_pb,k_size_ab,"Conv_fwd","MP_fwd","act_fwd","Conv_bwd","MP_bwd", "act_bwd");
 std::string str_op_size = integration_dims_to_string3(op_size_cf,op_size_pf,op_size_af, op_size_cb,op_size_pb, op_size_ab,"Conv_fwd","MP_fwd","act_fwd","Conv_bwd","MP_bwd","act_bwd");

  comp_conv_fwd<float>(testConvolutionSizes, srcDataConv.gpu(),
                                filterData.gpu(), NULL, dstDataGPU.gpu(),&avg_time1);
  compute_act_fwd(test_case1, dstDataGPU.gpu(), dataDst_act.gpu(),&avg_time2);
  comp_pool_fwd<float>(pool, dataDst_act.gpu(), dstData.gpu(), &avg_time3);
  comp_pool_bwd<float>(test_case, dataDst_act.gpu(), gradData1.gpu(), dstData.gpu(), &avg_time4);
  compute_hipdnn_act_bwd(test_case1, dataDst_act.gpu(), dataGrad_act.gpu(), dstData.gpu(), &avg_time5);
  comp_conv_bwd_filter<float>(testConvolutionSizes2, dataDst_act.gpu(),
                                 filterData.gpu(), gradData2.gpu(), NULL,
                                 dataGrad_act.gpu(),&avg_time6);

  avg_time = avg_time1 + avg_time2 + avg_time3 + avg_time4 + avg_time5 + avg_time6;
   
  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;


    std::string strt = "./result_unittest.csv";
    std::string testname = "convolution_pooling_act_fwd_bwd_intg: func_check_naive_conv_pool_act_fwd_bwd";
    float* temp = gradData2.getDataFromGPU();
    std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());
    write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size); 
}
