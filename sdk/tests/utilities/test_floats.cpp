// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>

TEST(TestHalf, BasicUsage)
{
    half h = 1.0_h;
    EXPECT_EQ(h, 1.0_h);
}

TEST(TestHalf, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_h), 1.0_h);
    EXPECT_EQ(std::fabs(1.0_h), 1.0_h);
}

TEST(TestHalf, Max)
{
    half a = 1.0_h;
    half b = 2.0_h;
    EXPECT_EQ(std::max(a, b), 2.0_h);
    EXPECT_EQ(std::max(b, a), 2.0_h);
}

TEST(TestBFloat16, BasicUsage)
{
    hip_bfloat16 bf = 1.0_bf;
    EXPECT_EQ(bf, 1.0_bf);
}

TEST(TestBFloat16, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_bf), 1.0_bf);
    EXPECT_EQ(std::fabs(1.0_bf), 1.0_bf);
}

TEST(TestBFloat16, Max)
{
    hip_bfloat16 a = 1.0_bf;
    hip_bfloat16 b = 2.0_bf;
    EXPECT_EQ(std::max(a, b), 2.0_bf);
    EXPECT_EQ(std::max(b, a), 2.0_bf);
}
