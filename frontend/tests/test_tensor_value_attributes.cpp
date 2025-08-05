// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <limits>
#include <numbers>
#include <vector>

using namespace hipdnn_frontend::graph;
using hipdnn_frontend::DataType_t;
using namespace hipdnn_sdk::data_objects;

TEST(TensorValueAttributesTests, SetGetClearFloat)
{
    Tensor_attributes tensor;
    EXPECT_FALSE(tensor.has_value());

    constexpr float test_value = std::numbers::pi_v<float>;
    tensor.set_value(test_value);
    EXPECT_TRUE(tensor.has_value());

    auto opt = tensor.get_value<float>();
    ASSERT_TRUE(opt.has_value());
    EXPECT_FLOAT_EQ(opt.value(), test_value);

    tensor.clear_value();
    EXPECT_FALSE(tensor.has_value());
    EXPECT_FALSE(tensor.get_value<float>().has_value());
}

TEST(TensorValueAttributesTests, PackUnpackFloat)
{
    Tensor_attributes tensor;
    tensor.set_uid(7)
        .set_name("value_tensor")
        .set_data_type(DataType_t::FLOAT)
        .set_stride({1, 2})
        .set_dim({3, 4})
        .set_is_virtual(false)
        .set_value(std::numbers::e_v<float>);

    flatbuffers::FlatBufferBuilder builder;
    auto fb_offset = tensor.pack_attributes(builder);
    builder.Finish(fb_offset);

    auto buffer_pointer = builder.GetBufferPointer();
    auto fb_tensor = flatbuffers::GetRoot<TensorAttributes>(buffer_pointer);

    EXPECT_EQ(fb_tensor->uid(), 7);
    EXPECT_STREQ(fb_tensor->name()->c_str(), "value_tensor");
    EXPECT_EQ(fb_tensor->data_type(), DataType_FLOAT);
    EXPECT_EQ(fb_tensor->strides()->size(), 2u);
    EXPECT_EQ(fb_tensor->dims()->size(), 2u);
    EXPECT_FALSE(fb_tensor->virtual_());

    EXPECT_EQ(fb_tensor->value_type(), TensorValue_Float32Value);
    auto fval = fb_tensor->value_as_Float32Value();
    ASSERT_NE(fval, nullptr);
    EXPECT_FLOAT_EQ(fval->value(), std::numbers::e_v<float>);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fb_tensor->UnPack());
    EXPECT_EQ(unpacked->uid, 7);
    EXPECT_EQ(unpacked->name, "value_tensor");
    EXPECT_EQ(unpacked->data_type, DataType_FLOAT);

    std::vector<int64_t> expected_strides = {1, 2};
    std::vector<int64_t> expected_dims = {3, 4};
    EXPECT_EQ(unpacked->strides, expected_strides);
    EXPECT_EQ(unpacked->dims, expected_dims);

    EXPECT_FALSE(unpacked->virtual_);

    ASSERT_EQ(unpacked->value.type, TensorValue_Float32Value);
    auto* float_val = unpacked->value.AsFloat32Value();
    ASSERT_NE(float_val, nullptr);
    EXPECT_FLOAT_EQ(float_val->value(), std::numbers::e_v<float>);
}

TEST(TensorValueAttributesTests, PackUnpackHalf)
{
    Tensor_attributes tensor;
    tensor.set_uid(8)
        .set_name("half_tensor")
        .set_data_type(DataType_t::HALF)
        .set_stride({1, 2})
        .set_dim({3, 4})
        .set_is_virtual(false)
        .set_value(uint16_t{16384});

    flatbuffers::FlatBufferBuilder builder;
    auto fb_offset = tensor.pack_attributes(builder);
    builder.Finish(fb_offset);

    auto buffer_pointer = builder.GetBufferPointer();
    auto fb_tensor = flatbuffers::GetRoot<TensorAttributes>(buffer_pointer);

    EXPECT_EQ(fb_tensor->value_type(), TensorValue_Float16Value);
    auto hval = fb_tensor->value_as_Float16Value();
    ASSERT_NE(hval, nullptr);
    EXPECT_EQ(hval->value(), uint16_t{16384});

    auto unpacked = std::unique_ptr<TensorAttributesT>(fb_tensor->UnPack());
    ASSERT_EQ(unpacked->value.type, TensorValue_Float16Value);
    auto* half_val = unpacked->value.AsFloat16Value();
    ASSERT_NE(half_val, nullptr);
    EXPECT_EQ(half_val->value(), uint16_t{16384});
}

TEST(TensorValueAttributesTests, PackUnpackDouble)
{
    Tensor_attributes tensor;
    tensor.set_uid(9)
        .set_name("double_tensor")
        .set_data_type(DataType_t::DOUBLE)
        .set_stride({1, 2})
        .set_dim({3, 4})
        .set_is_virtual(false)
        .set_value(std::numbers::pi_v<double>);

    flatbuffers::FlatBufferBuilder builder;
    auto fb_offset = tensor.pack_attributes(builder);
    builder.Finish(fb_offset);

    auto buffer_pointer = builder.GetBufferPointer();
    auto fb_tensor = flatbuffers::GetRoot<TensorAttributes>(buffer_pointer);

    EXPECT_EQ(fb_tensor->value_type(), TensorValue_Float64Value);
    auto dval = fb_tensor->value_as_Float64Value();
    ASSERT_NE(dval, nullptr);
    EXPECT_DOUBLE_EQ(dval->value(), std::numbers::pi_v<double>);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fb_tensor->UnPack());
    ASSERT_EQ(unpacked->value.type, TensorValue_Float64Value);
    auto* double_val = unpacked->value.AsFloat64Value();
    ASSERT_NE(double_val, nullptr);
    EXPECT_DOUBLE_EQ(double_val->value(), std::numbers::pi_v<double>);
}

TEST(TensorValueAttributesTests, PackUnpackEmptyValue)
{
    Tensor_attributes tensor;
    tensor.set_uid(10)
        .set_name("empty_tensor")
        .set_data_type(DataType_t::FLOAT)
        .set_stride({1, 2})
        .set_dim({3, 4})
        .set_is_virtual(true);

    EXPECT_FALSE(tensor.has_value());

    flatbuffers::FlatBufferBuilder builder;
    auto fb_offset = tensor.pack_attributes(builder);
    builder.Finish(fb_offset);

    auto buffer_pointer = builder.GetBufferPointer();
    auto fb_tensor = flatbuffers::GetRoot<TensorAttributes>(buffer_pointer);

    EXPECT_EQ(fb_tensor->value_type(), TensorValue_NONE);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fb_tensor->UnPack());
    EXPECT_EQ(unpacked->value.type, TensorValue_NONE);
}

TEST(TensorValueAttributesTests, TypeSafety)
{
    Tensor_attributes tensor;
    tensor.set_value(42.0f);

    auto float_opt = tensor.get_value<float>();
    ASSERT_TRUE(float_opt.has_value());
    EXPECT_FLOAT_EQ(float_opt.value(), 42.0f);

    EXPECT_FALSE(tensor.get_value<uint16_t>().has_value());
    EXPECT_FALSE(tensor.get_value<uint8_t>().has_value());
    EXPECT_FALSE(tensor.get_value<int32_t>().has_value());
    EXPECT_FALSE(tensor.get_value<double>().has_value());

    tensor.set_value(int32_t{123});

    EXPECT_FALSE(tensor.get_value<float>().has_value());

    auto int_opt = tensor.get_value<int32_t>();
    ASSERT_TRUE(int_opt.has_value());
    EXPECT_EQ(int_opt.value(), 123);
}

TEST(TensorValueAttributesTests, NumericLimits)
{
    Tensor_attributes tensor;

    tensor.set_value(std::numeric_limits<float>::max());
    auto float_opt = tensor.get_value<float>();
    ASSERT_TRUE(float_opt.has_value());
    EXPECT_FLOAT_EQ(float_opt.value(), std::numeric_limits<float>::max());

    tensor.set_value(std::numeric_limits<int32_t>::min());
    auto int_opt = tensor.get_value<int32_t>();
    ASSERT_TRUE(int_opt.has_value());
    EXPECT_EQ(int_opt.value(), std::numeric_limits<int32_t>::min());

    tensor.set_value(std::numeric_limits<uint8_t>::max());
    auto uint8_opt = tensor.get_value<uint8_t>();
    ASSERT_TRUE(uint8_opt.has_value());
    EXPECT_EQ(uint8_opt.value(), std::numeric_limits<uint8_t>::max());

    tensor.set_value(std::numeric_limits<double>::infinity());

    flatbuffers::FlatBufferBuilder builder;
    auto fb_offset = tensor.pack_attributes(builder);
    builder.Finish(fb_offset);

    auto buffer_pointer = builder.GetBufferPointer();
    auto fb_tensor = flatbuffers::GetRoot<TensorAttributes>(buffer_pointer);

    auto dval = fb_tensor->value_as_Float64Value();
    ASSERT_NE(dval, nullptr);
    EXPECT_TRUE(std::isinf(dval->value()));
}
