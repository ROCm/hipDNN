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

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeMissingXTensor)
{
    Conv_fprop_attributes conv_attributes;

    // Set w tensor with proper dims and strides
    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    // Set y tensor with proper dims and strides
    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    // X tensor is missing
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeMissingWTensor)
{
    Conv_fprop_attributes conv_attributes;

    // Set x tensor with proper dims and strides
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    // Set y tensor with proper dims and strides
    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    // W tensor is missing
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeMissingYTensor)
{
    Conv_fprop_attributes conv_attributes;

    // Set x tensor with proper dims and strides
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    // Set w tensor with proper dims and strides
    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    // Y tensor is missing
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeMissingConvolutionParameters)
{
    Conv_fprop_attributes conv_attributes;

    // Set all tensors with proper dims and strides
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_y(y_tensor);

    // Convolution parameters are missing
    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeAllValuesSet)
{
    Conv_fprop_attributes conv_attributes;

    // Set all tensors with proper dims and strides
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
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
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
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
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
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
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
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
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
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

TEST(ConvolutionFwdNodeTests, StrideInferenceWithStride2x2)
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
    // No dimensions or strides set - should be inferred
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({2, 2}); // 2x2 stride
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 64); // Output channels
    EXPECT_EQ(inferred_dims[2], 16); // Height: (32 + 1 + 1 - 3) / 2 + 1 = 16
    EXPECT_EQ(inferred_dims[3], 16); // Width: (32 + 1 + 1 - 3) / 2 + 1 = 16

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);

    // Expected strides for 16x16 output with NCHW layout
    EXPECT_EQ(inferred_strides[0], 16384); // N stride: 64 * 16 * 16 = 16384
    EXPECT_EQ(inferred_strides[1], 256); // C stride: 16 * 16 = 256
    EXPECT_EQ(inferred_strides[2], 16); // H stride: 16
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
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

TEST(ConvolutionFwdNodeTests, StrideInferenceWithLargeKernel5x5)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 28, 28});
    x_tensor->set_stride({50176, 784, 28, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 64, 5, 5}); // 5x5 kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({2, 2});
    conv_attributes.set_post_padding({2, 2});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 128); // Output channels
    EXPECT_EQ(inferred_dims[2], 28); // Height: (28 + 2 + 2 - 5) / 1 + 1 = 28
    EXPECT_EQ(inferred_dims[3], 28); // Width: (28 + 2 + 2 - 5) / 1 + 1 = 28

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 100352); // N stride: 128 * 28 * 28 = 100352
    EXPECT_EQ(inferred_strides[1], 784); // C stride: 28 * 28 = 784
    EXPECT_EQ(inferred_strides[2], 28); // H stride: 28
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, StrideInferenceWithAsymmetricPadding)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({2, 32, 16, 16});
    x_tensor->set_stride({8192, 256, 16, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 32, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({0, 1}); // Asymmetric padding
    conv_attributes.set_post_padding({1, 0});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 2); // Batch size
    EXPECT_EQ(inferred_dims[1], 64); // Output channels
    EXPECT_EQ(inferred_dims[2], 15); // Height: (16 + 0 + 1 - 3) / 1 + 1 = 15
    EXPECT_EQ(inferred_dims[3], 15); // Width: (16 + 1 + 0 - 3) / 1 + 1 = 15

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 14400); // N stride: 64 * 15 * 15 = 14400
    EXPECT_EQ(inferred_strides[1], 225); // C stride: 15 * 15 = 225
    EXPECT_EQ(inferred_strides[2], 15); // H stride: 15
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, StrideInferenceWithDilation2x2)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 16, 20, 20});
    x_tensor->set_stride({6400, 400, 20, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({32, 16, 3, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({2, 2});
    conv_attributes.set_post_padding({2, 2});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({2, 2}); // 2x2 dilation

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 32); // Output channels
    // Effective kernel size with dilation: (3-1)*2 + 1 = 5
    EXPECT_EQ(inferred_dims[2], 20); // Height: (20 + 2 + 2 - 5) / 1 + 1 = 20
    EXPECT_EQ(inferred_dims[3], 20); // Width: (20 + 2 + 2 - 5) / 1 + 1 = 20

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 12800); // N stride: 32 * 20 * 20 = 10368
    EXPECT_EQ(inferred_strides[1], 400); // C stride: 20 * 20 = 400
    EXPECT_EQ(inferred_strides[2], 20); // H stride: 20
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, StrideInferenceWithStride3x3AndLargeKernel)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 224, 224});
    x_tensor->set_stride({150528, 50176, 224, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 7, 7}); // 7x7 kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({3, 3});
    conv_attributes.set_post_padding({3, 3});
    conv_attributes.set_stride({3, 3}); // 3x3 stride
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 64); // Output channels
    EXPECT_EQ(inferred_dims[2], 75); // Height: (224 + 3 + 3 - 7) / 3 + 1 = 75
    EXPECT_EQ(inferred_dims[3], 75); // Width: (224 + 3 + 3 - 7) / 3 + 1 = 75

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 360000); // N stride: 64 * 75 * 75 = 360000
    EXPECT_EQ(inferred_strides[1], 5625); // C stride: 75 * 75 = 5625
    EXPECT_EQ(inferred_strides[2], 75); // H stride: 75
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, StrideInferenceWith1x1ConvolutionNoPadding)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 256, 14, 14});
    x_tensor->set_stride({50176, 196, 14, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({512, 256, 1, 1}); // 1x1 kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({0, 0}); // No padding
    conv_attributes.set_post_padding({0, 0});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 512); // Output channels
    EXPECT_EQ(inferred_dims[2], 14); // Height: (14 + 0 + 0 - 1) / 1 + 1 = 14
    EXPECT_EQ(inferred_dims[3], 14); // Width: (14 + 0 + 0 - 1) / 1 + 1 = 14

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 100352); // N stride: 512 * 14 * 14 = 100352
    EXPECT_EQ(inferred_strides[1], 196); // C stride: 14 * 14 = 196
    EXPECT_EQ(inferred_strides[2], 14); // H stride: 14
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, StrideInference3DConvolutionWithDilation)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 32, 16, 16, 16});
    x_tensor->set_stride({131072, 4096, 256, 16, 1}); // NCDHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 32, 3, 3, 3}); // 3x3x3 kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1, 1});
    conv_attributes.set_post_padding({1, 1, 1});
    conv_attributes.set_stride({1, 1, 1});
    conv_attributes.set_dilation({2, 1, 1}); // Dilation only in depth dimension

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 5);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 64); // Output channels
    EXPECT_EQ(inferred_dims[2], 14); // Depth: (16 + 1 + 1 - 5) / 1 + 1 = 14
    EXPECT_EQ(inferred_dims[3], 16); // Height: (16 + 1 + 1 - 3) / 1 + 1 = 16
    EXPECT_EQ(inferred_dims[4], 16); // Width: (16 + 1 + 1 - 3) / 1 + 1 = 16

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 5);
    EXPECT_EQ(inferred_strides[0], 229376); // N stride: 64 * 14 * 16 * 16 = 229376
    EXPECT_EQ(inferred_strides[1], 3584); // C stride: 14 * 16 * 16 = 3584
    EXPECT_EQ(inferred_strides[2], 256); // D stride: 16 * 16 = 256
    EXPECT_EQ(inferred_strides[3], 16); // H stride: 16
    EXPECT_EQ(inferred_strides[4], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, StrideInferenceWithNHWCLayoutAndComplexParams)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({2, 56, 56, 128});
    x_tensor->set_stride({401408, 1, 7168, 56}); // NHWC layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({56, 56, 5, 5}); // 5x5 kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 2});
    conv_attributes.set_post_padding({2, 1});
    conv_attributes.set_stride({2, 2});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 2); // Batch size
    EXPECT_EQ(inferred_dims[1], 56); // Output channels
    EXPECT_EQ(inferred_dims[2], 28); // Height: (56 + 2 + 1 - 5) / 2 + 1 = 28
    EXPECT_EQ(inferred_dims[3], 64); // Width: (128 + 2 + 1 - 5) / 2 + 1 = 64

    // Check inferred strides (should maintain NHWC order)
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_GT(inferred_strides[0], inferred_strides[2]); // N stride > H stride
    EXPECT_GT(inferred_strides[2], inferred_strides[3]); // H stride > W stride
    EXPECT_GT(inferred_strides[3], inferred_strides[1]); // W stride > C stride
    EXPECT_EQ(inferred_strides[1], 1); // C stride should be 1 (contiguous)

    EXPECT_EQ(inferred_strides[0], 100352); // N stride: 56 * 28 * 64 = 100352
    EXPECT_EQ(inferred_strides[1], 1); // C stride: 1
    EXPECT_EQ(inferred_strides[2], 3584); // H stride: 56 * 64 = 3584
    EXPECT_EQ(inferred_strides[3], 56); // W stride: 56
}

TEST(ConvolutionFwdNodeTests, StrideInferenceDepthwiseConvolution)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 32, 112, 112});
    x_tensor->set_stride({401408, 12544, 112, 1}); // NCHW layout
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({32, 32, 3, 3}); // Depthwise 3x3 kernel
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({2, 2});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 32); // Output channels (same as input for depthwise)
    EXPECT_EQ(inferred_dims[2], 56); // Height: (112 + 1 + 1 - 3) / 2 + 1 = 56
    EXPECT_EQ(inferred_dims[3], 56); // Width: (112 + 1 + 1 - 3) / 2 + 1 = 56

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 100352); // N stride: 32 * 56 * 56 = 100352
    EXPECT_EQ(inferred_strides[1], 3136); // C stride: 56 * 56 = 3136
    EXPECT_EQ(inferred_strides[2], 56); // H stride: 56
    EXPECT_EQ(inferred_strides[3], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, PreValidateTensorDimsEmpty)
{
    Conv_fprop_attributes conv_attributes;

    // Test with empty input dimensions
    auto x_tensor = std::make_shared<Tensor_attributes>();
    // No dimensions set
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

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateTensorDimsTooFew)
{
    Conv_fprop_attributes conv_attributes;

    // Test with too few dimensions (less than 3)
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3}); // Only 2 dimensions
    x_tensor->set_stride({3, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1});
    conv_attributes.set_post_padding({1});
    conv_attributes.set_stride({1});
    conv_attributes.set_dilation({1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateStrideMismatchInput)
{
    Conv_fprop_attributes conv_attributes;

    // Test with mismatched stride count
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32}); // Missing one stride
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateWeightDimsMismatch)
{
    Conv_fprop_attributes conv_attributes;

    // Test with mismatched weight dimensions
    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3, 3}); // 5D weight for 4D input
    w_tensor->set_stride({81, 27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateOutputStridesWithoutDims)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    // Set strides without dimensions
    y_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateSpatialParamMismatch)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32, 32}); // 3D spatial
    x_tensor->set_stride({98304, 32768, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3, 3});
    w_tensor->set_stride({81, 27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    // Only 2 spatial parameters for 3D spatial dimensions
    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1, 1});
    conv_attributes.set_stride({1, 1, 1});
    conv_attributes.set_dilation({1, 1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateGroupedConvInputChannelNotDivisible)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 32, 32}); // 64 input channels
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 48, 3, 3}); // 48 is not a divisor of 64
    w_tensor->set_stride({432, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateGroupedConvOutputChannelNotDivisible)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 32, 32});
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 32, 3, 3}); // Valid grouped setup
    w_tensor->set_stride({288, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 96, 32, 32}); // 96 is not a multiple of 64
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesGroupedConv2Groups)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 32, 32}); // 64 input channels
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 32, 3, 3}); // 32 input channels per group, 128 output channels
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

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 128); // Output channels
    EXPECT_EQ(inferred_dims[2], 32); // Height
    EXPECT_EQ(inferred_dims[3], 32); // Width

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 131072); // N stride: 128 * 32 * 32
    EXPECT_EQ(inferred_strides[1], 1024); // C stride: 32 * 32
    EXPECT_EQ(inferred_strides[2], 32); // H stride 32
    EXPECT_EQ(inferred_strides[3], 1); // W stride 1
}

TEST(ConvolutionFwdNodeTests, InferPropertiesGroupedConv4Groups)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({2, 128, 16, 16}); // 128 input channels
    x_tensor->set_stride({32768, 256, 16, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 32, 5, 5}); // 32 input channels per group (4 groups)
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({2, 2});
    conv_attributes.set_post_padding({2, 2});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 2); // Batch size
    EXPECT_EQ(inferred_dims[1], 64); // Output channels
    EXPECT_EQ(inferred_dims[2], 16); // Height
    EXPECT_EQ(inferred_dims[3], 16); // Width

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 16384); // N stride: 64 * 16 * 16
    EXPECT_EQ(inferred_strides[1], 256); // C stride: 16 * 16
    EXPECT_EQ(inferred_strides[2], 16); // H stride
    EXPECT_EQ(inferred_strides[3], 1); // W stride
}

TEST(ConvolutionFwdNodeTests, InferPropertiesGroupedConvWithStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 96, 112, 112}); // 96 input channels
    x_tensor->set_stride({1204224, 12544, 112, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({256, 48, 3, 3}); // 48 input channels per group (2 groups)
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({2, 2}); // Stride 2x2
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 256); // Output channels: 256
    EXPECT_EQ(inferred_dims[2], 56); // Height: (112 + 1 + 1 - 3) / 2 + 1 = 56
    EXPECT_EQ(inferred_dims[3], 56); // Width

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_EQ(inferred_strides[0], 802816); // N stride: 256 * 56 * 56
    EXPECT_EQ(inferred_strides[1], 3136); // C stride: 56 * 56
    EXPECT_EQ(inferred_strides[2], 56); // H stride
    EXPECT_EQ(inferred_strides[3], 1); // W stride
}

TEST(ConvolutionFwdNodeTests, InferPropertiesGroupedConvNHWCLayout)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 28, 28}); // NHWC layout, 64 input channels
    x_tensor->set_stride({50176, 1, 1792, 64}); // NHWC strides
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 16, 28, 28}); // 16 input channels per group (4 groups)
    w_tensor->set_stride({12544, 1, 448, 16}); // NHWC strides for weights
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

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 4);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 128); // Output channels
    EXPECT_EQ(inferred_dims[2], 3); // Height: (28 + 1 + 1 - 28) / 1 + 1 = 3
    EXPECT_EQ(inferred_dims[3], 3); // Width: (28 + 1 + 1 - 28) / 1 + 1 = 3

    // Check inferred strides maintain NHWC layout
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 4);
    EXPECT_GT(inferred_strides[0], inferred_strides[2]); // N > H
    EXPECT_EQ(inferred_strides[1], 1); // C should be contiguous
    EXPECT_GT(inferred_strides[2], inferred_strides[3]); // H > W
    EXPECT_GT(inferred_strides[3], inferred_strides[1]); // W > C
}

TEST(ConvolutionFwdNodeTests, InferPropertiesGroupedConv3D)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 48, 8, 16, 16}); // 48 input channels
    x_tensor->set_stride({98304, 2048, 256, 16, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({96, 16, 3, 3, 3}); // 16 input channels per group (3 groups)
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1, 1});
    conv_attributes.set_post_padding({1, 1, 1});
    conv_attributes.set_stride({1, 1, 1});
    conv_attributes.set_dilation({1, 1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 5);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 96); // Output channels
    EXPECT_EQ(inferred_dims[2], 8); // Depth
    EXPECT_EQ(inferred_dims[3], 16); // Height
    EXPECT_EQ(inferred_dims[4], 16); // Width

    // Check inferred strides
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 5);
    EXPECT_EQ(inferred_strides[0], 196608); // N stride: 96 * 8 * 16 * 16
    EXPECT_EQ(inferred_strides[1], 2048); // C stride: 8 * 16 * 16
    EXPECT_EQ(inferred_strides[2], 256); // D stride: 16 * 16 = 256
    EXPECT_EQ(inferred_strides[3], 16); // H stride: 16
    EXPECT_EQ(inferred_strides[4], 1); // W stride: 1
}

TEST(ConvolutionFwdNodeTests, InferPropertiesGroupedConv3DNHWCLayout)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 48, 8, 16, 16}); // NDHWC layout, 48 input channels
    x_tensor->set_stride({98304, 1, 12288, 768, 48}); // NDHWC strides
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({96, 16, 3, 3, 3}); // 16 input channels per group (3 groups)
    w_tensor->set_stride({432, 1, 144, 48, 16}); // NDHWC strides for weights
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1, 1});
    conv_attributes.set_post_padding({1, 1, 1});
    conv_attributes.set_stride({1, 1, 1});
    conv_attributes.set_dilation({1, 1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferred_dims = y_tensor->get_dim();
    EXPECT_EQ(inferred_dims.size(), 5);
    EXPECT_EQ(inferred_dims[0], 1); // Batch size
    EXPECT_EQ(inferred_dims[1], 96); // Output channels
    EXPECT_EQ(inferred_dims[2], 8); // Depth: (8 + 1 + 1 - 3) / 1 + 1 = 8
    EXPECT_EQ(inferred_dims[3], 16); // Height: (16 + 1 + 1 - 3) / 1 + 1 = 16
    EXPECT_EQ(inferred_dims[4], 16); // Width: (16 + 1 + 1 - 3) / 1 + 1 = 16

    // Check inferred strides maintain NDHWC layout
    auto inferred_strides = y_tensor->get_stride();
    EXPECT_EQ(inferred_strides.size(), 5);
    EXPECT_GT(inferred_strides[0], inferred_strides[2]); // N > D
    EXPECT_EQ(inferred_strides[1], 1); // C should be contiguous
    EXPECT_GT(inferred_strides[2], inferred_strides[3]); // D > H
    EXPECT_GT(inferred_strides[3], inferred_strides[4]); // H > W
    EXPECT_GT(inferred_strides[4], inferred_strides[1]); // W > C
}

TEST(ConvolutionFwdNodeTests, PreValidateOutputDimsMismatchBatch)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({2, 64, 32, 32});
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 64, 3, 3});
    w_tensor->set_stride({576, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 128, 32, 32}); // Wrong batch size
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateOutputDimsWrongCount)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 32, 32});
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 64, 3, 3});
    w_tensor->set_stride({576, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 128, 32, 32, 1}); // 5D output for 4D input
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeNegativeStrideValues)
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
    conv_attributes.set_stride({-1, 1}); // Negative stride in first dimension
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeNegativeDilationValues)
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
    conv_attributes.set_dilation({1, -1}); // Negative dilation in second dimension

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeNegativePrePaddingValues)
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

    conv_attributes.set_pre_padding({-1, 1}); // Negative pre-padding in first dimension
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeNegativePostPaddingValues)
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
    conv_attributes.set_post_padding({1, -2}); // Negative post-padding
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativePrePadding)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({-1, 1}); // Negative padding
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativePostPadding)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, -2}); // Negative post padding
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({0, 1}); // Zero stride
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, -1}); // Negative stride
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroDilation)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 0}); // Zero dilation

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeDilation)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({-2, 1}); // Negative dilation

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeBoundaryValueZeroPadding)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    // Zero padding is valid (boundary case)
    conv_attributes.set_pre_padding({0, 0});
    conv_attributes.set_post_padding({0, 0});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroInputDimension)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 0, 32, 32}); // Zero channel dimension
    x_tensor->set_stride({0, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeInputDimension)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, -32, 32}); // Negative height dimension
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroInputStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 0, 32, 1}); // Zero stride for channel dimension
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeInputStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, -32, 1}); // Negative stride
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroWeightDimension)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({0, 3, 3, 3}); // Zero output channels
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroWeightInputChannels)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 32, 32});
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({128, 0, 3, 3}); // Zero input channels - would cause division by zero
    w_tensor->set_stride({0, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeWeightDimension)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, -3, 3}); // Negative kernel height
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroWeightStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 0, 1}); // Zero stride
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeWeightStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({-27, 9, 3, 1}); // Negative stride
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroOutputDimension)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 0, 32}); // Zero height dimension
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeOutputDimension)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({-1, 64, 32, 32}); // Negative batch dimension
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeZeroOutputStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({0, 1024, 32, 1}); // Zero batch stride
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeNegativeOutputStride)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 3, 32, 32});
    x_tensor->set_stride({3072, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({64, 3, 3, 3});
    w_tensor->set_stride({27, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    y_tensor->set_dim({1, 64, 32, 32});
    y_tensor->set_stride({65536, 1024, 32, -1}); // Negative width stride
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(ConvolutionFwdNodeTests, PreValidateGroupedConvInvalidOutputChannels)
{
    Conv_fprop_attributes conv_attributes;

    auto x_tensor = std::make_shared<Tensor_attributes>();
    x_tensor->set_dim({1, 64, 32, 32}); // 64 input channels
    x_tensor->set_stride({65536, 1024, 32, 1});
    conv_attributes.set_x(x_tensor);

    auto w_tensor = std::make_shared<Tensor_attributes>();
    w_tensor->set_dim({63, 32, 3, 3}); // 32 input channels per group, 2 groups, 63 output channels
    w_tensor->set_stride({288, 9, 3, 1});
    conv_attributes.set_w(w_tensor);

    auto y_tensor = std::make_shared<Tensor_attributes>();
    conv_attributes.set_y(y_tensor);

    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}
