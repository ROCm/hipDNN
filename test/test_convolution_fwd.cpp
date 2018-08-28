#include "gtest/gtest.h"
#include "test_convolution_fwd_common.hpp"

TEST(convolution_fwd, func_check_zero_padding) {
    Desc inputDesc(1, 3, 224, 224);
    Desc filterDesc(21, 3, 3, 3);
    int pad[2]={0, 0}; // zero padding
    int stride[2]={1, 1};  // stride 1
    Desc outputDesc = calculateConv2DOutputDesc(inputDesc, filterDesc, pad, stride);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstDataCPU = createMemory<float>(outputDesc);
    Memory<float> dstDataGPU = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
    test_convolution_sizes_t testConvolutionSizes(inputDesc.N, 1, inputDesc.C,
            inputDesc.H, inputDesc.W, outputDesc.C, outputDesc.H, outputDesc.W,
            filterDesc.H, filterDesc.W, pad[0], pad[1], stride[0], stride[1], 1, 1);

    compute_cpuref_conv_fwd<float>(testConvolutionSizes, srcData.cpu(), filterData.cpu(), NULL, dstDataCPU.cpu());
    compute_hipdnn_conv_fwd<float>(testConvolutionSizes, srcData.gpu(), filterData.gpu(), NULL, dstDataGPU.gpu());
    Equals<float>(dstDataCPU, dstDataGPU);
	EXPECT_EQ(0, 0);
}
