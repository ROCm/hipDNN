#ifndef TEST_CONVOLUTION_BACKWARD_FILTER_HPP
#define TEST_CONVOLUTION_BACKWARD_FILTER_HPP

#include "hipDNN_test_common.h"

void print2(const float *data, int n, int c, int h, int w) {
    std::vector<float> buffer(1 << 20);

    HIP_CALL(hipMemcpy(buffer.data(), data, n * c * h * w * sizeof(float),
                       hipMemcpyDeviceToHost));

    int a = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << "n=" << n << ", c=" << c << ":" << std::endl;
            for (int k = 0; k < h; ++k) {
                for (int l = 0; l < w; ++l) {
                    std::cout << "\t" << std::setw(4) << std::right
                              << buffer[a];
                    ++a;
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

Desc calculateConv2DOutputDesc_bwd(Desc inputDesc, Desc filterDesc, int pad[2],
                                   int stride[2]) {
    assert(inputDesc.C == filterDesc.C);
    int outputHeight =
        ((inputDesc.H - filterDesc.H + 2 * pad[0]) / stride[0]) + 1;
    int outputWidth =
        ((inputDesc.W - filterDesc.W + 2 * pad[1]) / stride[1]) + 1;
    Desc outputDesc(inputDesc.N, filterDesc.N, outputHeight, outputWidth);
    return outputDesc;
}

struct test_convolution_bwd_filter {
    test_convolution_bwd_filter(int mb, int ng, int ic, int ih, int iw, int oc,
                                int oh, int ow, int kh, int kw, int padh,
                                int padw, int strh, int strw, int dilh = 0,
                                int dilw = 0)
        : mb(mb), ng(ng), ic(ic), ih(ih), iw(iw), oc(oc), oh(oh), ow(ow),
          kh(kh), kw(kw), padh(padh), padw(padw), strh(strh), strw(strw),
          dilh(dilh), dilw(dilw) {}
    int mb;         // mini batches
    int ng;         // number of groups
    int ic, ih, iw; // Input channels, height and width
    int oc, oh, ow; // Output channels, height and width
    int kh, kw;     // kernel height and width
    int padh, padw; // padding along height and width
    int strh, strw; // stride along height and width
    int dilh, dilw; // dilation along height and width
};

template <typename T>
__global__ void dev_const(hipLaunchParm lp, T *px, float k) {
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    px[tid] = k;
}

template <typename dataType>
void compute_hipdnn_conv_bwd_filter(test_convolution_bwd_filter &c,
                                    dataType *src, dataType *weights,
                                    dataType *grad, dataType *bias,
                                    dataType *dst, double &avg_time) {

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
    hipdnnConvolutionFwdAlgo_t algo;
    int MaxAlgoCount = 2;
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

    hipLaunchKernel(dev_const, c.mb * c.oc, c.oh * c.ow, 0, 0, dst, 0.0f);

    float alpha = 1.f;
    float beta = 0.f;

    checkHIPDNN(hipdnnConvolutionForward(
        hipdnn, &alpha, in_desc, src, filt_desc, weights, conv_desc, algo,
        ws_data, ws_size, &beta, out_desc, dst));

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
        std::uint64_t time_elapsed = timer.elapsed_nanoseconds();
        time_vector[i] = (double)time_elapsed / 1000;
    }

    avg_time = std::accumulate(time_vector.begin() + 10, time_vector.end(), 0) /
               (benchmark_iterations - 10);
    std::cout << "#######################################Average Time: "
              << avg_time << std::endl;
    // std::cout<<"\nweights aftr back_conv:\n";
    // print2(weights,c.oc,c.ic,c.kh,c.kw);

    // std::cout<<"\ngrad aftr back_conv:\n";

    // print2(grad,c.oc,c.ic,c.kh,c.kw);

    // std::cout<<"\ndst aftr back_conv:\n";

    // print2(dst,c.mb,c.oc,c.oh,c.ow);

    hipFree(ws_data);
    hipdnnDestroyTensorDescriptor(out_desc);
    hipdnnDestroyConvolutionDescriptor(conv_desc);
    hipdnnDestroyFilterDescriptor(filt_desc);
    hipdnnDestroyTensorDescriptor(in_desc);
    hipdnnDestroy(hipdnn);
}

#endif // TEST_CONVOLUTION_FORWARD_FILTER_HPP
