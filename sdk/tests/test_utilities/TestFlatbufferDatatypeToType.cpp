// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <type_traits>

#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeToType.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

// Compile-time checks
static_assert(std::is_same_v<DataTypeToNative<DataType::FLOAT>::type, float>);
static_assert(std::is_same_v<DataTypeToNative<DataType::HALF>::type, half>);
static_assert(std::is_same_v<DataTypeToNative<DataType::DOUBLE>::type, double>);
static_assert(std::is_same_v<DataTypeToNative<DataType::INT32>::type, int32_t>);
static_assert(std::is_same_v<DataTypeToNative<DataType::BFLOAT16>::type, hip_bfloat16>);

TEST(TestFlatbufferDatatypeToType, RuntimeSizeChecks)
{
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::FLOAT>::type), sizeof(float));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::HALF>::type), sizeof(half));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::DOUBLE>::type), sizeof(double));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::INT32>::type), sizeof(int32_t));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::BFLOAT16>::type), sizeof(hip_bfloat16));
}

TEST(TestFlatbufferDatatypeToType, QualifierPropagation)
{
    // Ensure adding const/volatile works with type traits (sanity)
    using T = DataTypeToNative<DataType::FLOAT>::type;
    static_assert(std::is_same_v<std::add_const_t<T>, const float>);
    SUCCEED();
}
