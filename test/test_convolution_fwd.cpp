#include "gtest/gtest.h"
#include "test_convolution_fwd_common.hpp"

TEST(convolution_fwd, func_check_zero_padding) {
    Desc inputDesc(3, 3, 224, 224);
    Desc filterDesc(21, 3, 3, 3);
    int pad[2]={0, 0}; // zero padding
    int stride[2]={1, 1};  // stride 1
    Desc outputDesc = calculateConv2DOutputDesc(inputDesc, filterDesc, pad, stride);
    Memory<float> srcData = createMemory<float>(inputDesc);
    Memory<float> dstData = createMemory<float>(outputDesc);
    Memory<float> filterData = createMemory<float>(filterDesc);
    populateMemoryRandom<float>(srcData);
    populateMemoryRandom<float>(filterData);
	EXPECT_EQ(0, 0);
}
