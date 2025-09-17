// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <type_traits>

#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeToType.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

// Compile-time checks
static_assert(std::is_same_v<DataTypeToNative_t<DataType_FLOAT>, float>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType_HALF>, half>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType_DOUBLE>, double>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType_INT32>, int32_t>);
static_assert(std::is_same_v<DataTypeToNative_t<DataType_BFLOAT16>, hip_bfloat16>);

TEST(TestFlatbufferDatatypeToType, RuntimeSizeChecks)
{
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType_FLOAT>), sizeof(float));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType_HALF>), sizeof(half));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType_DOUBLE>), sizeof(double));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType_INT32>), sizeof(int32_t));
    EXPECT_EQ(sizeof(DataTypeToNative_t<DataType_BFLOAT16>), sizeof(hip_bfloat16));
}

TEST(TestFlatbufferDatatypeToType, QualifierPropagation)
{
    // Ensure adding const/volatile works with type traits (sanity)
    using T = DataTypeToNative_t<DataType_FLOAT>;
    static_assert(std::is_same_v<std::add_const_t<T>, const float>);
    SUCCEED();
}
