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
    auto packed = tensor.pack_attributes(builder);
    builder.Finish(packed);

    auto buffer_pointer = builder.GetBufferPointer();
    auto tensor_attributes_flatbuffer
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(buffer_pointer);
    auto unpacked = std::unique_ptr<hipdnn_sdk::data_objects::TensorAttributesT>(
        tensor_attributes_flatbuffer->UnPack());

    EXPECT_EQ(unpacked->uid, 1);
    EXPECT_EQ(unpacked->name, "PackedTensor");
    EXPECT_EQ(unpacked->data_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(unpacked->strides, std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(unpacked->dims, std::vector<int64_t>({4, 5, 6}));
    EXPECT_TRUE(unpacked->virtual_);
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveValidCase)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 6, 1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveEmptyDims)
{
    Tensor_attributes tensor;
    // No dimensions set
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveZeroDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 0, 6}); // Zero in middle dimension
    tensor.set_stride({0, 6, 1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveNegativeDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, -5, 6}); // Negative dimension
    tensor.set_stride({30, 6, 1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveZeroStride)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 0, 1}); // Zero stride
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveNegativeStride)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, -6, 1}); // Negative stride
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveStrideSizeMismatch)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 6}); // Only 2 strides for 3 dimensions
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveMoreStridesThanDims)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5});
    tensor.set_stride({20, 5, 1}); // 3 strides for 2 dimensions
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveAllZeroDims)
{
    Tensor_attributes tensor;
    tensor.set_dim({0, 0, 0});
    tensor.set_stride({0, 0, 0});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveAllNegativeDims)
{
    Tensor_attributes tensor;
    tensor.set_dim({-1, -2, -3});
    tensor.set_stride({6, 3, 1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveMixedInvalidValues)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 0, -6}); // Mix of valid, zero, and negative
    tensor.set_stride({0, -6, 1}); // Mix of zero, negative, and valid
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveSingleDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({10});
    tensor.set_stride({1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveSingleZeroDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({0});
    tensor.set_stride({1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositiveLargeDimensions)
{
    Tensor_attributes tensor;
    tensor.set_dim({1024, 2048, 4096});
    tensor.set_stride({8388608, 4096, 1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsAndStridesSetAndPositive5DValidCase)
{
    Tensor_attributes tensor;
    tensor.set_dim({2, 3, 4, 5, 6});
    tensor.set_stride({360, 120, 30, 6, 1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveValidCase)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveEmptyDims)
{
    Tensor_attributes tensor;
    // No dimensions set
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveZeroDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 0, 6}); // Zero in middle dimension
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveNegativeDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, -5, 6}); // Negative dimension
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveAllZeroDims)
{
    Tensor_attributes tensor;
    tensor.set_dim({0, 0, 0});
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveAllNegativeDims)
{
    Tensor_attributes tensor;
    tensor.set_dim({-1, -2, -3});
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveMixedInvalidValues)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 0, -6}); // Mix of valid, zero, and negative
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveSingleDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({10});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveSingleZeroDimension)
{
    Tensor_attributes tensor;
    tensor.set_dim({0});
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveLargeDimensions)
{
    Tensor_attributes tensor;
    tensor.set_dim({1024, 2048, 4096});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositive5DValidCase)
{
    Tensor_attributes tensor;
    tensor.set_dim({2, 3, 4, 5, 6});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TensorAttributesTests, ValidateDimsSetAndPositiveWithStridesIgnored)
{
    Tensor_attributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({0, -1, 2}); // Invalid strides should be ignored
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}
