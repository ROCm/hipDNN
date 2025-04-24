// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TensorAttributesTests, DefaultConstructor)
{
    Tensor_attributes tensor;
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_EQ(tensor.get_name(), "");
    EXPECT_EQ(tensor.get_data_type(), DataType_t::NOT_SET);
    EXPECT_TRUE(tensor.get_stride().empty());
    EXPECT_TRUE(tensor.get_dim().empty());
    EXPECT_EQ(tensor.get_volume(), 1);
    EXPECT_FALSE(tensor.get_is_virtual());
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TensorAttributesTests, SetAndGetUid)
{
    Tensor_attributes tensor;
    tensor.set_uid(42);
    EXPECT_EQ(tensor.get_uid(), 42);
    EXPECT_TRUE(tensor.has_uid());

    tensor.clear_uid();
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TensorAttributesTests, SetAndGetName)
{
    Tensor_attributes tensor;
    tensor.set_name("TestTensor");
    EXPECT_EQ(tensor.get_name(), "TestTensor");
}

TEST(TensorAttributesTests, SetAndGetDataType)
{
    Tensor_attributes tensor;
    tensor.set_data_type(DataType_t::FLOAT);
    EXPECT_EQ(tensor.get_data_type(), DataType_t::FLOAT);
}

TEST(TensorAttributesTests, SetAndGetStride)
{
    Tensor_attributes tensor;
    tensor.set_stride({1, 2, 3});
    EXPECT_EQ(tensor.get_stride(), std::vector<int64_t>({1, 2, 3}));
}

TEST(TensorAttributesTests, SetAndGetDim)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    EXPECT_EQ(tensor.get_dim(), std::vector<int64_t>({4, 5, 6}));
    EXPECT_EQ(tensor.get_volume(), 4 * 5 * 6);
}

TEST(TensorAttributesTests, SetAndGetIsVirtual)
{
    Tensor_attributes tensor;
    tensor.set_is_virtual(true);
    EXPECT_TRUE(tensor.get_is_virtual());

    tensor.set_is_virtual(false);
    EXPECT_FALSE(tensor.get_is_virtual());
}

TEST(TensorAttributesTests, SetOutput)
{
    Tensor_attributes tensor;
    tensor.set_output(true);
    EXPECT_FALSE(tensor.get_is_virtual());

    tensor.set_output(false);
    EXPECT_TRUE(tensor.get_is_virtual());
}

TEST(TensorAttributesTests, SetFromGraphAttributes)
{
    Graph_attributes graph_attributes;
    graph_attributes.set_io_data_type(DataType_t::FLOAT);
    graph_attributes.set_intermediate_data_type(DataType_t::HALF);

    Tensor_attributes tensor;
    tensor.set_is_virtual(false).set_from_graph_attributes(graph_attributes);
    EXPECT_EQ(tensor.get_data_type(), DataType_t::FLOAT);

    tensor.set_data_type(DataType_t::NOT_SET);
    tensor.set_is_virtual(true).set_from_graph_attributes(graph_attributes);
    EXPECT_EQ(tensor.get_data_type(), DataType_t::HALF);
}

TEST(TensorAttributesTests, PackAttributes)
{
    Tensor_attributes tensor;
    tensor.set_uid(1)
        .set_name("PackedTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_stride({1, 2, 3})
        .set_dim({4, 5, 6})
        .set_is_virtual(true);

    flatbuffers::FlatBufferBuilder builder;
    auto                           packed = tensor.pack_attributes(builder);
    builder.Finish(packed);

    auto buffer_pointer = builder.GetBufferPointer();
    auto tensor_attributes_flatbuffer
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(buffer_pointer);
    auto unpacked = tensor_attributes_flatbuffer->UnPack();

    EXPECT_EQ(unpacked->uid, 1);
    EXPECT_EQ(unpacked->name, "PackedTensor");
    EXPECT_EQ(unpacked->data_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(unpacked->strides, std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(unpacked->dims, std::vector<int64_t>({4, 5, 6}));
    EXPECT_TRUE(unpacked->virtual_);
}