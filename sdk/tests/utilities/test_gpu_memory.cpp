// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/gpu_memory.hpp>

using namespace hipdnn::sdk::utilities;

class GpuMemoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }
};

TEST_F(GpuMemoryTest, AllocationAndDeallocation)
{
    constexpr size_t num_elements = 10;
    Gpu_memory<float> gpu_mem(num_elements);
    EXPECT_NE(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), num_elements);
}

TEST_F(GpuMemoryTest, ZeroElements)
{
    Gpu_memory<double> gpu_mem(0);
    EXPECT_EQ(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), 0);
}

TEST_F(GpuMemoryTest, WriteAndReadBack)
{
    constexpr size_t num_elements = 4;
    Gpu_memory<int> gpu_mem(num_elements);

    std::vector<int> host_data = {1, 2, 3, 4};
    ASSERT_EQ(host_data.size(), num_elements);

    // Copy to device
    ASSERT_EQ(
        hipMemcpy(
            gpu_mem.data(), host_data.data(), num_elements * sizeof(int), hipMemcpyHostToDevice),
        hipSuccess);

    // Copy back to host
    std::vector<int> host_out(num_elements, 0);
    ASSERT_EQ(
        hipMemcpy(
            host_out.data(), gpu_mem.data(), num_elements * sizeof(int), hipMemcpyDeviceToHost),
        hipSuccess);

    EXPECT_EQ(host_data, host_out);
}

TEST_F(GpuMemoryTest, DimsConstructor1D)
{
    std::vector<int64_t> dims = {7};
    Gpu_memory<float> gpu_mem(dims);
    EXPECT_NE(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), 7);
}

TEST_F(GpuMemoryTest, DimsConstructor2D)
{
    std::vector<int64_t> dims = {3, 5};
    Gpu_memory<int> gpu_mem(dims);
    EXPECT_NE(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), 15);
}

TEST_F(GpuMemoryTest, DimsConstructor3D)
{
    std::vector<int64_t> dims = {2, 3, 4};
    Gpu_memory<double> gpu_mem(dims);
    EXPECT_NE(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), 24);
}

TEST_F(GpuMemoryTest, DimsConstructorZeroDim)
{
    std::vector<int64_t> dims = {};
    Gpu_memory<float> gpu_mem(dims);
    EXPECT_NE(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), 1);
}

TEST_F(GpuMemoryTest, DimsConstructorZeroInDims)
{
    std::vector<int64_t> dims = {2, 0, 4};
    Gpu_memory<float> gpu_mem(dims);
    EXPECT_EQ(gpu_mem.data(), nullptr);
    EXPECT_EQ(gpu_mem.size(), 0);
}
