// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/HalfUtils.hpp>
#include <hipdnn_sdk/utilities/HipBfloat16Utils.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::utilities;

template <typename T>
MigratableMemory<T> createBuffer(size_t size, T mult)
{
    MigratableMemory<T> buffer(size);

    T* data = buffer.hostData();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<T>(static_cast<float>(i)) * mult;
    }

    return buffer;
}

TEST(CpuFpReferenceValidation, BasicBFloat16Usage)
{
    CpuFpReferenceValidation<hip_bfloat16> refValidation;

    auto buffer1 = createBuffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = createBuffer<hip_bfloat16>(100, 1.0_bf);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BasicHalfUsage)
{
    CpuFpReferenceValidation<half> refValidation;

    auto buffer1 = createBuffer<half>(100, 1.0_h);
    auto buffer2 = createBuffer<half>(100, 1.0_h);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BasicFloatUsage)
{
    CpuFpReferenceValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(100, 1.0f);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BasicDoubleUsage)
{
    CpuFpReferenceValidation<double> refValidation;

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 1.0);

    EXPECT_TRUE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BFloat16NotComparable)
{
    CpuFpReferenceValidation<hip_bfloat16> refValidation;

    auto buffer1 = createBuffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = createBuffer<hip_bfloat16>(100, 2.0_bf);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, HalfNotComparable)
{
    CpuFpReferenceValidation<half> refValidation;

    auto buffer1 = createBuffer<half>(100, 1.0_h);
    auto buffer2 = createBuffer<half>(100, 2.0_h);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, FloatNotComparable)
{
    CpuFpReferenceValidation<float> refValidation;

    auto buffer1 = createBuffer<float>(100, 1.0f);
    auto buffer2 = createBuffer<float>(100, 2.0f);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, DoubleNotComparable)
{
    CpuFpReferenceValidation<double> refValidation;

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 2.0);

    EXPECT_FALSE(refValidation.allClose(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, ToleranceComparison)
{
    CpuFpReferenceValidation<double> refValidationLowTolerance(1e-7, 1e-7);
    CpuFpReferenceValidation<double> refValidationHighTolerance(1e-5, 1e-5);

    auto buffer1 = createBuffer<double>(100, 1.0);
    auto buffer2 = createBuffer<double>(100, 1.000001);

    // Set a very small tolerance
    EXPECT_TRUE(refValidationHighTolerance.allClose(buffer1, buffer2));

    // Change the tolerance to a larger value
    EXPECT_FALSE(refValidationLowTolerance.allClose(buffer1, buffer2));
}
