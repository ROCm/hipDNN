#ifndef TEST_CONVOLUTION_COMMON_HPP
#define TEST_CONVOLUTION_COMMON_HPP

#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename dataType>
void compute_hipdnn_conv_forward(convulution_Size &c, dataType *src,
            dataType *weights, dataType *bias, dataType *dst, float alpha,
            float beta, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));

  hipdnnFilterDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
  int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
  checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                          HIPDNN_TENSOR_NCHW, 4, filterDimA));

  hipdnnConvolutionDescriptor_t conv_desc;
  checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));

  checkHIPDNN(hipdnnSetConvolution2dDescriptor(
      conv_desc, c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
      HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
      conv_desc, in_desc, filt_desc, &c.mb, &c.oc, &c.oh, &c.ow));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.oc, c.oh, c.ow));

  hipdnnConvolutionFwdAlgo_t algo;

  int MaxAlgoCount = 1;
  size_t ws_size{0};
  float *ws_data{nullptr};
  int calgo;

  hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];

  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(
     hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  hipMalloc(&ws_data, ws_size);

  hipdnnFindConvolutionForwardAlgorithmEx(
      hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst,
      MaxAlgoCount, &calgo, algoPerf, ws_data, ws_size);

  algo = (hipdnnConvolutionFwdAlgo_t)algoPerf[0].algo;

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        checkHIPDNN(hipdnnConvolutionForward(hipdnn, &alpha, in_desc, src,
                                  filt_desc, weights, conv_desc, algo, ws_data,
                                  ws_size, &beta, out_desc, dst));

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                           time_vector.end(), 0) / (benchmark_iterations - 10);

  hipFree(ws_data);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);

}

template <typename dataType>
void compute_hipdnn_conv_backward_filter(convulution_Size &c, dataType *src,
                             dataType *weights, dataType *grad, dataType *bias,
                             dataType *dst, float alpha, float beta,
                             float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));

  hipdnnFilterDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
  int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
  checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                          HIPDNN_TENSOR_NCHW, 4, filterDimA));

  hipdnnConvolutionDescriptor_t conv_desc;
  checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
  checkHIPDNN(hipdnnSetConvolution2dDescriptor(
      conv_desc, c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
      HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc, &c.mb, &c.oc, &c.oh, &c.ow));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.oc, c.oh, c.ow));

  int MaxAlgoCount = 2;
  size_t ws_size{0};
  float *ws_data{nullptr};
  int calgo;

  hipdnnConvolutionBwdFilterAlgo_t  b_algo=HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  ws_size =0;

  hipdnnConvolutionBwdFilterAlgoPerf_t b_algoPerf[MaxAlgoCount];

  checkHIPDNN(hipdnnGetConvolutionBackwardFilterWorkspaceSize(
                hipdnn, in_desc, out_desc, conv_desc, filt_desc,
                b_algo, &ws_size));

  hipMalloc(&ws_data, ws_size);

  hipdnnFindConvolutionBackwardFilterAlgorithmEx(hipdnn, in_desc, src, out_desc,
                              dst, conv_desc, filt_desc, weights, MaxAlgoCount ,
                              &calgo, b_algoPerf, ws_data, ws_size);

  b_algo = (hipdnnConvolutionBwdFilterAlgo_t)b_algoPerf[0].algo;

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        checkHIPDNN(hipdnnConvolutionBackwardFilter(hipdnn, &alpha, in_desc,
                                                  src, out_desc, dst, conv_desc,
                                                  b_algo, ws_data, ws_size,
                                                  &beta,filt_desc,grad ));

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                            time_vector.end(), 0) / (benchmark_iterations - 10);

  hipFree(ws_data);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

template <typename dataType>
void compute_hipdnn_conv_backward_data(convulution_Size &c, dataType *src,
                             dataType *weights, dataType *grad, dataType *bias,
                             dataType *dst, float *avg_time) {

  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));

  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih, c.iw));

  hipdnnFilterDescriptor_t filt_desc;
  checkHIPDNN(hipdnnCreateFilterDescriptor(&filt_desc));
  int filterDimA[] = {c.oc, c.ic, c.kh, c.kw};
  checkHIPDNN(hipdnnSetFilterNdDescriptor(filt_desc, HIPDNN_DATA_FLOAT,
                                          HIPDNN_TENSOR_NCHW, 4, filterDimA));

  hipdnnConvolutionDescriptor_t conv_desc;
  checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv_desc));
  checkHIPDNN(hipdnnSetConvolution2dDescriptor(
      conv_desc, c.padh, c.padw, c.strh, c.strw, c.dilh, c.dilw,
      HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(
      conv_desc, in_desc, filt_desc, &c.mb, &c.oc, &c.oh, &c.ow));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, c.mb, c.oc, c.oh, c.ow));

  hipdnnConvolutionFwdAlgo_t algo;
  algo = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  int MaxAlgoCount = 5;
  size_t ws_size = 0;
  float *ws_data;
  int calgo;

  hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];

  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(
      hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  hipMalloc(&ws_data, ws_size);

  hipdnnFindConvolutionForwardAlgorithmEx(
      hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst,
      MaxAlgoCount, &calgo, algoPerf, ws_data, ws_size);

  algo = (hipdnnConvolutionFwdAlgo_t)algoPerf[0].algo;

  float alpha = 1.f;
  float beta = 0.f;

  checkHIPDNN(hipdnnConvolutionForward(hipdnn, &alpha, in_desc, src, filt_desc,
                                       weights, conv_desc, algo, ws_data,
                                       ws_size, &beta, out_desc, dst));

  hipdnnConvolutionBwdDataAlgo_t algo_bd = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  ws_size =0;

  hipdnnConvolutionBwdDataAlgoPerf_t algoPerf_bd[MaxAlgoCount];

  checkHIPDNN(hipdnnGetConvolutionBackwardDataWorkspaceSize(hipdnn, filt_desc,
                              out_desc, conv_desc, in_desc, algo_bd, &ws_size));

  hipMalloc(&ws_data, ws_size);

  hipdnnFindConvolutionBackwardDataAlgorithmEx(hipdnn, filt_desc, weights,
                          out_desc, dst, conv_desc, in_desc, src, MaxAlgoCount,
                          &calgo, algoPerf_bd, ws_data, ws_size);

  algo_bd = algoPerf_bd[0].algo;

  high_resolution_timer_t timer;

  std::vector<double> time_vector(benchmark_iterations, 0);

  for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();

        checkHIPDNN(hipdnnConvolutionBackwardData( hipdnn, &alpha, filt_desc,
                               weights, out_desc, dst, conv_desc, algo_bd,
                               ws_data, ws_size, &beta, in_desc, grad));

        hipDeviceSynchronize();

        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();

        time_vector[i] = (double)time_elapsed / 1000;
	  }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10,
                            time_vector.end(), 0) / (benchmark_iterations - 10);

  hipFree(ws_data);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

template <typename dataType>
void Test_convolution_fwd(Desc inputDesc, Desc filterDesc, int pad[2],
            int stride[2], int dil[2], std::string testname, int alpha, int beta)
{
  float avg_time = 0;
  Desc outputDesc = calculate_Dims(inputDesc, filterDesc, pad, stride,dil);

  Memory<float> srcData = createMemory<float>(inputDesc);
  Memory<float> dstDataGPU = createMemory<float>(outputDesc);
  Memory<float> filterData = createMemory<float>(filterDesc);

  populateMemoryRandom<float>(srcData);
  populateMemoryRandom<float>(filterData);

  convulution_Size testConvolutionSizes(
        inputDesc.N, 1, inputDesc.C, inputDesc.H, inputDesc.W, outputDesc.C,
        outputDesc.H, outputDesc.W, filterDesc.H, filterDesc.W, pad[0], pad[1],
        stride[0], stride[1], dil[0], dil[1]);

  compute_hipdnn_conv_forward<float>(testConvolutionSizes, srcData.gpu(),
                            filterData.gpu(), NULL, dstDataGPU.gpu(), alpha,
                            beta, &avg_time);

  std::cout << "\nAverage Time is: " << avg_time << "micro seconds"<<std::endl;

  std::string strt = "./result_unittest.csv";
  std::string filename="convolution_forward.csv";

  int ip_size[4] = {inputDesc.N, inputDesc.C, inputDesc.H, inputDesc.W};
  int k_size[4] = {filterDesc.N, filterDesc.C, filterDesc.H, filterDesc.W};
  int op_size[4] =  {outputDesc.N, outputDesc.C, outputDesc.H, outputDesc.W};

  std::string str_ip_size  = convert_to_string((int*)ip_size,4);
  std::string str_k_size  = convert_to_string((int*)k_size,4);
  std::string str_op_size  = convert_to_string((int*)op_size,4);

  float* temp = dstDataGPU.getDataFromGPU();

  std::string str  = convert_to_string((float*)temp,(int)dstDataGPU.get_num_elements());

  write_to_csv(strt, str, testname, avg_time, str_ip_size, str_k_size, str_op_size);
  dump_result_csv(filename, testname, temp, (int)dstDataGPU.get_num_elements());

}

#endif // TEST_CONVOLUTION_COMMON_H
