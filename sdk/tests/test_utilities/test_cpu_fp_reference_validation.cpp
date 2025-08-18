// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>

using namespace hipdnn_sdk::reference_test_utilities;
using namespace hipdnn_sdk::utilities;

template <typename T>
Migratable_memory create_buffer(size_t size, T mult)
{
    Migratable_memory buffer(size, sizeof(T));

    T* data = buffer.host_data<T>();

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<T>(static_cast<float>(i)) * mult;
    }

    return buffer;
}

TEST(CpuFpReferenceValidation, BasicBFloat16Usage)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<hip_bfloat16> ref_validation;

    auto buffer1 = create_buffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = create_buffer<hip_bfloat16>(100, 1.0_bf);

    EXPECT_TRUE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BasicHalfUsage)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<half> ref_validation;

    auto buffer1 = create_buffer<half>(100, 1.0_h);
    auto buffer2 = create_buffer<half>(100, 1.0_h);

    EXPECT_TRUE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BasicFloatUsage)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<float> ref_validation;

    auto buffer1 = create_buffer<float>(100, 1.0f);
    auto buffer2 = create_buffer<float>(100, 1.0f);

    EXPECT_TRUE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BasicDoubleUsage)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<double> ref_validation;

    auto buffer1 = create_buffer<double>(100, 1.0);
    auto buffer2 = create_buffer<double>(100, 1.0);

    EXPECT_TRUE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, BFloat16NotComparable)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<hip_bfloat16> ref_validation;

    auto buffer1 = create_buffer<hip_bfloat16>(100, 1.0_bf);
    auto buffer2 = create_buffer<hip_bfloat16>(100, 2.0_bf);

    EXPECT_FALSE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, HalfNotComparable)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<half> ref_validation;

    auto buffer1 = create_buffer<half>(100, 1.0_h);
    auto buffer2 = create_buffer<half>(100, 2.0_h);

    EXPECT_FALSE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, FloatNotComparable)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<float> ref_validation;

    auto buffer1 = create_buffer<float>(100, 1.0f);
    auto buffer2 = create_buffer<float>(100, 2.0f);

    EXPECT_FALSE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, DoubleNotComparable)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<double> ref_validation;

    auto buffer1 = create_buffer<double>(100, 1.0);
    auto buffer2 = create_buffer<double>(100, 2.0);

    EXPECT_FALSE(ref_validation.compare_buffers(buffer1, buffer2));
}

TEST(CpuFpReferenceValidation, ToleranceComparison)
{
    SKIP_IF_NO_DEVICES();
    Cpu_fp_reference_validation<double> ref_validation_low_tolerance(1e-7, 1e-7);
    Cpu_fp_reference_validation<double> ref_validation_high_tolerance(1e-5, 1e-5);

    auto buffer1 = create_buffer<double>(100, 1.0);
    auto buffer2 = create_buffer<double>(100, 1.000001);

    // Set a very small tolerance
    EXPECT_TRUE(ref_validation_high_tolerance.compare_buffers(buffer1, buffer2));

    // Change the tolerance to a larger value
    EXPECT_FALSE(ref_validation_low_tolerance.compare_buffers(buffer1, buffer2));
}
