// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/convolution_fwd_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/convolution_fprop_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(ConvolutionFwdNodeTests, PreValidateNode)
{
    Conv_fprop_attributes conv_attributes;
    conv_attributes.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeMissingValues)
{
    Conv_fprop_attributes conv_attributes;

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    // Test missing x tensor
    conv_attributes.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});
    auto conv_attributes_copy = conv_attributes;
    ConvolutionNode node_without_x(std::move(conv_attributes_copy), graph_attributes);

    error = node_without_x.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    // Test missing w tensor
    Conv_fprop_attributes conv_attributes2;
    conv_attributes2.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes2.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes2.set_pre_padding({1, 1});
    conv_attributes2.set_post_padding({1, 1});
    conv_attributes2.set_stride({1, 1});
    conv_attributes2.set_dilation({1, 1});
    ConvolutionNode node_without_w(std::move(conv_attributes2), graph_attributes);

    error = node_without_w.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    // Test missing y tensor
    Conv_fprop_attributes conv_attributes3;
    conv_attributes3.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes3.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes3.set_pre_padding({1, 1});
    conv_attributes3.set_post_padding({1, 1});
    conv_attributes3.set_stride({1, 1});
    conv_attributes3.set_dilation({1, 1});
    ConvolutionNode node_without_y(std::move(conv_attributes3), graph_attributes);

    error = node_without_y.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    // Test missing convolution parameters
    Conv_fprop_attributes conv_attributes4;
    conv_attributes4.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes4.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes4.set_y(std::make_shared<Tensor_attributes>());
    ConvolutionNode node_without_params(std::move(conv_attributes4), graph_attributes);

    error = node_without_params.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    // Test with all values
    Conv_fprop_attributes conv_attributes5;
    conv_attributes5.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes5.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes5.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes5.set_pre_padding({1, 1});
    conv_attributes5.set_post_padding({1, 1});
    conv_attributes5.set_stride({1, 1});
    conv_attributes5.set_dilation({1, 1});
    ConvolutionNode node_with_all_values(std::move(conv_attributes5), graph_attributes);

    error = node_with_all_values.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeMissingXTensor)
{
    Conv_fprop_attributes conv_attributes;
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeMissingWTensor)
{
    Conv_fprop_attributes conv_attributes;
    conv_attributes.set_x(std::make_shared<Tensor_attributes>());
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeMissingYTensor)
{
    Conv_fprop_attributes conv_attributes;
    conv_attributes.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes.set_w(std::make_shared<Tensor_attributes>());
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNode2DConvolutionSuccess)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 64); // Output channels
    EXPECT_EQ(inferred_dims[2], 32); // Height: (32 + 1 + 1 - 3) / 1 + 1 = 32
    EXPECT_EQ(inferred_dims[3], 32); // Width: (32 + 1 + 1 - 3) / 1 + 1 = 32
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNode3DConvolutionSuccess)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({2, 16, 8, 16, 16});
    x_tensor->set_stride({32768, 2048, 256, 16, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({32, 16, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({0, 1, 1});
    conv_attributes.set_post_padding({0, 1, 1});
    conv_attributes.set_stride({1, 1, 1});
    conv_attributes.set_dilation({1, 1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 5);
    EXPECT_EQ(inferred_dims[0], 2); // Batch size
    EXPECT_EQ(inferred_dims[1], 32); // Output channels
    EXPECT_EQ(inferred_dims[2], 6); // Depth: (8 + 0 + 0 - 3) / 1 + 1 = 6
    EXPECT_EQ(inferred_dims[3], 16); // Height: (16 + 1 + 1 - 3) / 1 + 1 = 16
    EXPECT_EQ(inferred_dims[4], 16); // Width: (16 + 1 + 1 - 3) / 1 + 1 = 16
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeInsufficientSpatialParameters)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1}); // Missing padding for second spatial dim
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeInvalidStrideValues)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({0, 1}); // Invalid stride
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeInvalidDilationValues)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 0}); // Invalid dilation

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeNegativeOutputSize)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 3, 3}); // Small input
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 5, 5}); // Large kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({0, 0});
    conv_attributes.set_post_padding({0, 0});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, StrideInferenceMissingInputStrides)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    // No stride set on input tensor
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32}); // Pre-set dimensions
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, StrideInferenceMissingOutputDimensions)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    y_tensor->set_dim({1}); // check that inferring fails when dims don't match x.
    EXPECT_EQ(node.infer_properties_node(), error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, StrideInferenceDimensionMismatch)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32}); // Missing one stride dimension
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, StrideInferenceNCHWLayoutSuccess)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    // No stride set - should be inferred
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    // Should maintain the same stride order as input (NCHW)
    EXPECT_GT(inferred_strides[1], inferred_strides[2]); // C stride > H stride
    EXPECT_GT(inferred_strides[2], inferred_strides[3]); // H stride > W stride
    EXPECT_EQ(inferred_strides[3], 1); // W stride should be 1 (contiguous)
}

TEST(ConvolutionFwdNodeTests, StrideInferenceNHWCLayoutSuccess)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 32, 32, 3});
    x_tensor->set_stride({3072, 96, 3, 1}); // NHWC layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 32, 32, 64});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    // Should maintain the same stride order as input (NHWC)
    EXPECT_GT(inferred_strides[0], inferred_strides[1]); // N stride > H stride
    EXPECT_GT(inferred_strides[1], inferred_strides[2]); // H stride > W stride
    EXPECT_GT(inferred_strides[2], inferred_strides[3]); // W stride > C stride
    EXPECT_EQ(inferred_strides[3], 1); // C stride should be 1 (contiguous)
}

TEST(ConvolutionFwdNodeTests, StrideInferencePreExistingStridesNotOverwritten)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({65536, 1024, 32, 1}); // Pre-existing strides
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto final_strides = y_tensor->get_stride();
    // Should keep the pre-existing strides
    EXPECT_EQ(final_strides[0], 65536);
    EXPECT_EQ(final_strides[1], 1024);
    EXPECT_EQ(final_strides[2], 32);
    EXPECT_EQ(final_strides[3], 1);
}

TEST(ConvolutionFwdNodeTests, PackNode)
{
    Conv_fprop_attributes conv_attributes;
    conv_attributes.name = "Convolution";

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_uid(2)
        .set_name("WTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({64, 3, 3, 3})
        .set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_uid(3)
        .set_name("YTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});
    conv_attributes.set_conv_mode(hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "Convolution");
    EXPECT_EQ(node_flatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes_ConvolutionFwdAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_ConvolutionFwdAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->x_tensor_uid(), x_tensor->get_uid());
    EXPECT_EQ(packed_attributes->w_tensor_uid(), w_tensor->get_uid());
    EXPECT_EQ(packed_attributes->y_tensor_uid(), y_tensor->get_uid());

    ASSERT_EQ(packed_attributes->pre_padding()->size(), 2);
    EXPECT_EQ(packed_attributes->pre_padding()->Get(0), 1);
    EXPECT_EQ(packed_attributes->pre_padding()->Get(1), 1);

    ASSERT_EQ(packed_attributes->post_padding()->size(), 2);
    EXPECT_EQ(packed_attributes->post_padding()->Get(0), 1);
    EXPECT_EQ(packed_attributes->post_padding()->Get(1), 1);

    ASSERT_EQ(packed_attributes->stride()->size(), 2);
    EXPECT_EQ(packed_attributes->stride()->Get(0), 1);
    EXPECT_EQ(packed_attributes->stride()->Get(1), 1);

    ASSERT_EQ(packed_attributes->dilation()->size(), 2);
    EXPECT_EQ(packed_attributes->dilation()->Get(0), 1);
    EXPECT_EQ(packed_attributes->dilation()->Get(1), 1);

    EXPECT_EQ(packed_attributes->conv_mode(), hipdnn_sdk::data_objects::ConvMode_CROSS_CORRELATION);
}

TEST(ConvolutionFwdNodeTests, GatherHipdnnTensorIds)
{
    Conv_fprop_attributes conv_attributes;
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_uid(1).set_name("XTensor");
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_uid(2).set_name("WTensor");
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_uid(3).set_name("YTensor");
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    std::unordered_set<int64_t> used_ids;
    node.gather_hipdnn_tensor_ids(used_ids);

    EXPECT_TRUE(used_ids.contains(1));
    EXPECT_TRUE(used_ids.contains(2));
    EXPECT_TRUE(used_ids.contains(3));
}

TEST(ConvolutionFwdNodeTests, PopulateHipdnnTensorIds)
{
    Conv_fprop_attributes conv_attributes;
    conv_attributes.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>> tensor_lookup;
    std::unordered_set<int64_t> used_ids;
    int64_t current_tensor_id = 1;

    auto error = node.populate_hipdnn_tensor_ids(tensor_lookup, current_tensor_id, used_ids);
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    std::vector<std::shared_ptr<Tensor_attributes>> tensors;
    tensors.reserve(node.attributes.inputs.size() + node.attributes.outputs.size());

    for(const auto& input_pair : node.attributes.inputs)
    {
        tensors.emplace_back(input_pair.second);
    }

    for(const auto& output_pair : node.attributes.outputs)
    {
        tensors.emplace_back(output_pair.second);
    }

    std::unordered_set<int64_t> tensor_ids;
    for(const auto& tensor : tensors)
    {
        ASSERT_TRUE(tensor->has_uid());
        EXPECT_TRUE(tensor_ids.insert(tensor->get_uid()).second)
            << "Duplicate tensor ID found: " << tensor->get_uid();
    }
}
