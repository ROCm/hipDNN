// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <numbers>
#include <vector>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

using namespace hipdnn_frontend::graph;
using hipdnn_frontend::DataType_t;
using namespace hipdnn_sdk::data_objects;

TEST(Tensor_value_attributes_tests, set_get_clear_float) {
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

TEST(Tensor_value_attributes_tests, pack_unpack_float) {
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

    EXPECT_EQ(fb_tensor->value_type(), Tensor_Value_FValue);
    auto fval = fb_tensor->value_as_FValue();
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
    
    ASSERT_EQ(unpacked->value.type, Tensor_Value_FValue);
    ASSERT_NE(unpacked->value.AsFValue(), nullptr);
    EXPECT_FLOAT_EQ(unpacked->value.AsFValue()->value, std::numbers::e_v<float>);
}

TEST(Tensor_value_attributes_tests, pack_unpack_half) {
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
    
    EXPECT_EQ(fb_tensor->value_type(), Tensor_Value_HValue);
    auto hval = fb_tensor->value_as_HValue();
    ASSERT_NE(hval, nullptr);
    EXPECT_EQ(hval->value(), uint16_t{16384});
    
    auto unpacked = std::unique_ptr<TensorAttributesT>(fb_tensor->UnPack());
    ASSERT_EQ(unpacked->value.type, Tensor_Value_HValue);
    ASSERT_NE(unpacked->value.AsHValue(), nullptr);
    EXPECT_EQ(unpacked->value.AsHValue()->value, uint16_t{16384});
}