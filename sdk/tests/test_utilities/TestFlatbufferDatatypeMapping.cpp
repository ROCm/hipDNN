// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <type_traits>

#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

// Compile-time checks for DataType to Native type mapping
static_assert(std::is_same_v<DataTypeToNative<DataType::FLOAT>::type, float>);
static_assert(std::is_same_v<DataTypeToNative<DataType::HALF>::type, half>);
static_assert(std::is_same_v<DataTypeToNative<DataType::DOUBLE>::type, double>);
static_assert(std::is_same_v<DataTypeToNative<DataType::INT32>::type, int32_t>);
static_assert(std::is_same_v<DataTypeToNative<DataType::BFLOAT16>::type, hip_bfloat16>);

// Compile-time checks for Native type to DataType mapping
static_assert(NativeToDataType<float>::value == DataType::FLOAT);
static_assert(NativeToDataType<half>::value == DataType::HALF);
static_assert(NativeToDataType<double>::value == DataType::DOUBLE);
static_assert(NativeToDataType<int32_t>::value == DataType::INT32);
static_assert(NativeToDataType<hip_bfloat16>::value == DataType::BFLOAT16);

TEST(TestFlatbufferDatatypeMapping, DataTypeToNativeRuntimeSizeChecks)
{
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::FLOAT>::type), sizeof(float));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::HALF>::type), sizeof(half));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::DOUBLE>::type), sizeof(double));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::INT32>::type), sizeof(int32_t));
    EXPECT_EQ(sizeof(DataTypeToNative<DataType::BFLOAT16>::type), sizeof(hip_bfloat16));
}

TEST(TestFlatbufferDatatypeMapping, NativeToDataTypeConversion)
{
    // Test that the reverse mapping produces correct DataType values
    EXPECT_EQ(NativeToDataType<float>::value, DataType::FLOAT);
    EXPECT_EQ(NativeToDataType<half>::value, DataType::HALF);
    EXPECT_EQ(NativeToDataType<double>::value, DataType::DOUBLE);
    EXPECT_EQ(NativeToDataType<int32_t>::value, DataType::INT32);
    EXPECT_EQ(NativeToDataType<hip_bfloat16>::value, DataType::BFLOAT16);
}

TEST(TestFlatbufferDatatypeMapping, BidirectionalConsistency)
{
    // Test that converting from DataType to native and back yields the same DataType
    constexpr auto FLOAT_TYPE = NativeToDataType<DataTypeToNative<DataType::FLOAT>::type>::value;
    static_assert(FLOAT_TYPE == DataType::FLOAT);
    EXPECT_EQ(FLOAT_TYPE, DataType::FLOAT);

    constexpr auto HALF_TYPE = NativeToDataType<DataTypeToNative<DataType::HALF>::type>::value;
    static_assert(HALF_TYPE == DataType::HALF);
    EXPECT_EQ(HALF_TYPE, DataType::HALF);

    constexpr auto DOUBLE_TYPE = NativeToDataType<DataTypeToNative<DataType::DOUBLE>::type>::value;
    static_assert(DOUBLE_TYPE == DataType::DOUBLE);
    EXPECT_EQ(DOUBLE_TYPE, DataType::DOUBLE);

    constexpr auto INT32_TYPE = NativeToDataType<DataTypeToNative<DataType::INT32>::type>::value;
    static_assert(INT32_TYPE == DataType::INT32);
    EXPECT_EQ(INT32_TYPE, DataType::INT32);

    constexpr auto BFLOAT16_TYPE
        = NativeToDataType<DataTypeToNative<DataType::BFLOAT16>::type>::value;
    static_assert(BFLOAT16_TYPE == DataType::BFLOAT16);
    EXPECT_EQ(BFLOAT16_TYPE, DataType::BFLOAT16);
}

TEST(TestFlatbufferDatatypeMapping, QualifierPropagation)
{
    // Ensure adding const/volatile works with type traits
    using FloatType = DataTypeToNative<DataType::FLOAT>::type;
    static_assert(std::is_same_v<std::add_const_t<FloatType>, const float>);
    static_assert(std::is_same_v<std::add_volatile_t<FloatType>, volatile float>);

    using HalfType = DataTypeToNative<DataType::HALF>::type;
    static_assert(std::is_same_v<std::add_const_t<HalfType>, const half>);

    SUCCEED();
}

TEST(TestFlatbufferDatatypeMapping, TypeIdentity)
{
    // Verify that the types are exactly what we expect
    using FloatResult = DataTypeToNative<DataType::FLOAT>::type;
    using HalfResult = DataTypeToNative<DataType::HALF>::type;
    using DoubleResult = DataTypeToNative<DataType::DOUBLE>::type;
    using Int32Result = DataTypeToNative<DataType::INT32>::type;
    using Bfloat16Result = DataTypeToNative<DataType::BFLOAT16>::type;

    // Runtime type checks using typeid
    EXPECT_EQ(typeid(FloatResult), typeid(float));
    EXPECT_EQ(typeid(HalfResult), typeid(half));
    EXPECT_EQ(typeid(DoubleResult), typeid(double));
    EXPECT_EQ(typeid(Int32Result), typeid(int32_t));
    EXPECT_EQ(typeid(Bfloat16Result), typeid(hip_bfloat16));
}
