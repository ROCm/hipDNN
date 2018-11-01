#ifndef TEST_CONVOLUTION_ACTIVATION_FORWARD_HPP
#define TEST_CONVOLUTION_ACTIVATION_FORWARD_HPP

#include "hipdnn_test_common.h"
#include "common.hpp"

template <typename T>
__global__ void dev_const(hipLaunchParm lp, T *px, float k) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    px[tid] = k;
}

template <typename dataType>
void compute_hipdnn_conv_fwd(convulution_Size &c, dataType *src,
                             dataType *weights, dataType *bias, dataType *dst, float *avg_time) {

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

  hipdnnFindConvolutionForwardAlgorithmEx(
      hipdnn, in_desc, src, filt_desc, weights, conv_desc, out_desc, dst,
      MaxAlgoCount, &calgo, algoPerf, ws_data, ws_size);
  algo = (hipdnnConvolutionFwdAlgo_t)algoPerf[0].algo;

  checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(
      hipdnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  hipMalloc(&ws_data, ws_size);

  float alpha = 1.f;
  float beta = 0.f;

    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        checkHIPDNN(hipdnnConvolutionForward(hipdnn, &alpha, in_desc, src, filt_desc,
                                       weights, conv_desc, algo, ws_data,
                                       ws_size, &beta, out_desc, dst));
     
        hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

  hipFree(ws_data);
  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyConvolutionDescriptor(conv_desc);
  hipdnnDestroyFilterDescriptor(filt_desc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

template <typename dataType>
void compute_hipdnn_activation_forward(activation_params_t &test_case,
                                        dataType *src,
                                        dataType *dst, float *avg_time) {
  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));
  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));
  hipdnnActivationDescriptor_t activationDesc;
  //hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_SIGMOID;
  hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_RELU;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;
  double reluCeilingOrAlpha = 0;
  double activBeta = 0;
  double activExp = 0;
  checkHIPDNN(hipdnnCreateActivationDescriptor(&activationDesc));
  checkHIPDNN(hipdnnSetActivationDescriptor(activationDesc, mode, reluNanOpt,
                                            reluCeilingOrAlpha, activBeta,
                                            activExp));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));
  float alpha = 1.f;
  float beta = 0.f;

   high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        hipdnnActivationForward(hipdnn, activationDesc, &alpha, in_desc, src, &beta,
                          out_desc, dst);

        hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);


  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyActivationDescriptor(activationDesc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

template <typename dataType>
void compute_conv_back_filter(convulution_Size &c,
                                    dataType *src, dataType *weights,
                                    dataType *grad, dataType *bias,
                                    dataType *dst, float *avg_time) {

    hipdnnHandle_t hipdnn;
    checkHIPDNN(hipdnnCreate(&hipdnn));

    hipdnnTensorDescriptor_t in_desc;
    checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
    checkHIPDNN(hipdnnSetTensor4dDescriptor(in_desc, HIPDNN_TENSOR_NCHW,
                                            HIPDNN_DATA_FLOAT, c.mb, c.ic, c.ih,
                                            c.iw));

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
    checkHIPDNN(hipdnnSetTensor4dDescriptor(out_desc, HIPDNN_TENSOR_NCHW,
                                            HIPDNN_DATA_FLOAT, c.mb, c.oc, c.oh,
                                            c.ow));
    int MaxAlgoCount = 2;
    size_t ws_size{0};
    float *ws_data{nullptr};
    int calgo;
    float alpha = 1.f;
    float beta = 0.f;

   hipdnnConvolutionBwdFilterAlgo_t b_algo =
        HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    ws_size = 0;
    hipdnnConvolutionBwdFilterAlgoPerf_t b_algoPerf[MaxAlgoCount];

    checkHIPDNN(hipdnnGetConvolutionBackwardFilterWorkspaceSize(
        hipdnn, in_desc, out_desc, conv_desc, filt_desc, b_algo, &ws_size));

    hipMalloc(&ws_data, ws_size);

    hipLaunchKernel(dev_const, c.oc * c.ic, c.kh * c.kw, 0, 0, grad, 0.0f);

    hipdnnFindConvolutionBackwardFilterAlgorithmEx(
        hipdnn, in_desc, src, out_desc, dst, conv_desc, filt_desc, weights,
        MaxAlgoCount, &calgo, b_algoPerf, ws_data, ws_size);
    b_algo = (hipdnnConvolutionBwdFilterAlgo_t)b_algoPerf[0].algo;

    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        checkHIPDNN(hipdnnConvolutionBackwardFilter(
            hipdnn, &alpha, in_desc, src, out_desc, dst, conv_desc, b_algo,
            ws_data, ws_size, &beta, filt_desc, grad));

        hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

    hipFree(ws_data);
    hipdnnDestroyTensorDescriptor(out_desc);
    hipdnnDestroyConvolutionDescriptor(conv_desc);
    hipdnnDestroyFilterDescriptor(filt_desc);
    hipdnnDestroyTensorDescriptor(in_desc);
    hipdnnDestroy(hipdnn);
}

template <typename dataType>
void compute_hipdnn_activation_bwd(activation_params_t &test_case,
                                        dataType *src, dataType *grad,
                                        dataType *dst, float *avg_time) {
  hipdnnHandle_t hipdnn;
  checkHIPDNN(hipdnnCreate(&hipdnn));
  hipdnnTensorDescriptor_t in_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&in_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));
  hipdnnActivationDescriptor_t activationDesc;
  //hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_SIGMOID;
  hipdnnActivationMode_t mode = HIPDNN_ACTIVATION_RELU;
  hipdnnNanPropagation_t reluNanOpt = HIPDNN_NOT_PROPAGATE_NAN;
  double reluCeilingOrAlpha = 0;
  double activBeta = 0;
  double activExp = 0;
  checkHIPDNN(hipdnnCreateActivationDescriptor(&activationDesc));
  checkHIPDNN(hipdnnSetActivationDescriptor(activationDesc, mode, reluNanOpt,
                                            reluCeilingOrAlpha, activBeta,
                                            activExp));

  hipdnnTensorDescriptor_t out_desc;
  checkHIPDNN(hipdnnCreateTensorDescriptor(&out_desc));
  checkHIPDNN(hipdnnSetTensor4dDescriptor(
      out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT, test_case.n,
      test_case.channels, test_case.height, test_case.width));
  float alpha = 1.f;
  float beta = 0.f;

    high_resolution_timer_t timer;
    std::vector<double> time_vector(benchmark_iterations, 0);
    for (int i = 0; i < benchmark_iterations; i++) {

        timer.restart();
        hipdnnActivationBackward(hipdnn, activationDesc, &alpha, in_desc, src,
                           in_desc, src, out_desc, dst, &beta, out_desc, grad);
        hipDeviceSynchronize();
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

  *avg_time = (float)std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) / (benchmark_iterations - 10);

  hipdnnDestroyTensorDescriptor(out_desc);
  hipdnnDestroyActivationDescriptor(activationDesc);
  hipdnnDestroyTensorDescriptor(in_desc);
  hipdnnDestroy(hipdnn);
}

#endif //TEST_CONVOLUTION_ACTIVATION_FORWARD_HPP
