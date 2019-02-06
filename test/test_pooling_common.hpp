#ifndef TEST_POOLING_COMMON_HPP
#define TEST_POOLING_COMMON_HPP

#include "hipdnn.h"
#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void hipdnn_pooling_forward(test_pooling_descriptor &c, dataType *src,
                            dataType *dst, hipdnnPoolingMode_t mode,
                            hipdnnDataType_t hipdataType, bool do_backward,
                            float alpha, float beta, float *avg_time) {

  hipdnnHandle_t handle;
  checkHIPDNN(hipdnnCreate(&handle));

  hipdnnTensorDescriptor_t in_desc, out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, hipdataType, c.mb, c.c, c.ih, c.iw));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));
  checkHIPDNN(hipdnnSetPooling2dDescriptor(pool_desc, mode,
                                           HIPDNN_NOT_PROPAGATE_NAN, c.kw, c.kh,
                                           c.padt, c.padl, c.strh, c.strw));

  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &c.mb,
                                                 &c.c, &c.oh, &c.ow));

  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, hipdataType, c.mb, c.c, c.oh, c.ow));

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        checkHIPDNN(hipdnnPoolingForward(handle, pool_desc, &alpha, in_desc, src,
                                   &beta, out_desc, dst, do_backward))

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                    0) / (benchmark_iterations - 10);

  checkHIPDNN(hipdnnDestroyTensorDescriptor(in_desc));
  checkHIPDNN(hipdnnDestroyTensorDescriptor(out_desc));
  checkHIPDNN(hipdnnDestroyPoolingDescriptor(pool_desc));
  checkHIPDNN(hipdnnDestroy(handle));

}

template <typename dataType>
void hipdnn_pooling_backward(test_pooling_descriptor &test_case, dataType *src,
                        dataType *grad, dataType *dst, hipdnnPoolingMode_t mode,
                        hipdnnDataType_t hipdataType, float alpha, float beta,
                        float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));

  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, hipdataType, test_case.mb,
      test_case.c, test_case.ih, test_case.iw));

  hipdnnPoolingDescriptor_t pool_desc;
  checkHIPDNN(hipdnnCreatePoolingDescriptor(&pool_desc));

  hipdnnNanPropagation_t maxpoolingNanOpt = HIPDNN_NOT_PROPAGATE_NAN;

  checkHIPDNN(hipdnnSetPooling2dDescriptor(
      pool_desc, mode, maxpoolingNanOpt, test_case.kh,
      test_case.kw, test_case.padt, test_case.padl,
      test_case.strh, test_case.strw));

  checkHIPDNN(hipdnnGetPooling2dForwardOutputDim(
      pool_desc, in_desc, &test_case.mb, &test_case.c,
      &test_case.oh, &test_case.ow))
  hipdnnTensorDescriptor_t out_desc;

  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));

  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, hipdataType, test_case.mb,
      test_case.c, test_case.oh, test_case.ow));

  hipdnnPoolingForward(hipdnn, pool_desc, &alpha, in_desc, src, &beta, out_desc,
                       dst, true);

  high_resolution_timer_t timer;
  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        hipdnnPoolingBackward(hipdnn, pool_desc, &alpha, out_desc, dst, out_desc,
                        dst, in_desc, src, &beta, in_desc, grad);

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(),
                                      0) / (benchmark_iterations - 10);

  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyPoolingDescriptor(pool_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void Test_pooling_fwd(Desc inputDesc, int spatial_ext[2], int stride[2],
           int pad[2], std::string testname, float alpha = 1.f, float beta = 0.f,
           hipdnnPoolingMode_t pool_mode = HIPDNN_POOLING_MAX,
           hipdnnDataType_t hipdataType = HIPDNN_DATA_FLOAT)
{

  float avg_time = 0;
  float* temp;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  Memory<dataType> srcData = createMemory<dataType>(inputDesc);
  Memory<dataType> dstDataGPU = createMemory<dataType>(outputDesc);
  populateMemoryRandom<dataType>(srcData);

  test_pooling_descriptor pool(inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W,
                               outputDesc.H, outputDesc.W, spatial_ext[0],
                               spatial_ext[1], pad[0], pad[1], stride[0],
                               stride[1]);

  hipdnn_pooling_forward<dataType>(pool, srcData.gpu(), dstDataGPU.gpu(), pool_mode,
                                hipdataType, false, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  int ip_size[4] = {pool.mb, pool.c, pool.ih, pool.iw};
  int k_size[4] = {pool.mb, pool.c, pool.kh, pool.kw};
  int op_size[4] =  {pool.mb, pool.c, pool.oh, pool.ow};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);


  if (hipdataType == HIPDNN_DATA_FLOAT)
  {
    temp = dstDataGPU.getDataFromGPU();
  }

  else
  {
    Memory<float> dstDataGPU_f(pool.mb * pool.c * pool.oh * pool.ow);
    Convert_toFloat<dataType>(dstDataGPU, dstDataGPU_f);
    temp = dstDataGPU_f.getDataFromGPU();
  }

  std::string strt = "./result_unittest.csv";
  std::string filename="pooling_forward.csv";

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());
}

template <typename dataType>
void Test_pooling_bwd(Desc inputDesc, int spatial_ext[2], int stride[2],
          int pad[2], std::string testname, float alpha = 1.f, float beta = 0.f,
          hipdnnPoolingMode_t pool_mode = HIPDNN_POOLING_MAX,
          hipdnnDataType_t hipdataType = HIPDNN_DATA_FLOAT)
{

  float avg_time = 0;
  float* temp;

  Desc outputDesc = calculate_pool_Dims(inputDesc, spatial_ext, pad, stride);

  test_pooling_descriptor test_case(inputDesc.N, inputDesc.C, inputDesc.H,
                        inputDesc.W, outputDesc.H, outputDesc.W, spatial_ext[0],
                        spatial_ext[1], pad[0], pad[1], stride[0], stride[1]);

  Memory<dataType> dataSrc = createMemory<dataType>(inputDesc);
  Memory<dataType> dataGrad = createMemory<dataType>(inputDesc);
  Memory<dataType> dataDst = createMemory<dataType>(outputDesc);

  populateMemoryRandom(dataSrc);

  hipdnn_pooling_backward<dataType>(test_case, dataSrc.gpu(), dataGrad.gpu(),
                 dataDst.gpu(), pool_mode, hipdataType, alpha, beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {inputDesc.N, inputDesc.C, spatial_ext[0], spatial_ext[1]};
  int op_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  if (hipdataType == HIPDNN_DATA_FLOAT)
  {
    temp = dataGrad.getDataFromGPU();
  }

  else
  {
    Memory<float> dstDataGPU_f(test_case.mb * test_case.c * test_case.ih * test_case.iw);
    Convert_toFloat<dataType>(dataGrad, dstDataGPU_f);
    temp = dstDataGPU_f.getDataFromGPU();
  }

  std::string str  = convert_to_string((float*)temp,
                                       (int)dataGrad.get_num_elements());

  std::string strt = "./result_unittest.csv";
  std::string filename="pooling_backward.csv";

  write_to_csv(strt, str, testname,avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dataGrad.get_num_elements());

}

#endif // TEST_POOLING_COMMON_H