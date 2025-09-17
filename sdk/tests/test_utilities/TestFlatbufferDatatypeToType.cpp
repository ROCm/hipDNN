// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <type_traits>

#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeToType.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

// Compile-time checks
static_assert(std::is_same_v<DataTypeToNative_t<DataType::FLOAT>, float>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType::HALF>, half>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType::DOUBLE>, double>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType::INT32>, int32_t>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType::BFLOAT16>, hip_bfloat16>);

TEST(TestFlatbufferDatatypeToType, RuntimeSizeChecks)
{
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType::FLOAT>), sizeof(float));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType::HALF>), sizeof(half));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType::DOUBLE>), sizeof(double));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType::INT32>), sizeof(int32_t));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType::BFLOAT16>), sizeof(hip_bfloat16));
}

TEST(TestFlatbufferDatatypeToType, QualifierPropagation)
{
    // Ensure adding const/volatile works with type traits (sanity)
    using T = DataTypeToNative_t<DataType::FLOAT>;
    static_assert(std::is_same_v<std::add_const_t<T>, const float>);
    SUCCEED();
}
