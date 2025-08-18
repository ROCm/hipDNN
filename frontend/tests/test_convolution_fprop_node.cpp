// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/convolution_fwd_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/convolution_fwd_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(ConvolutionFwdNodeTests, PreValidateNode)
{
    Convolution_fprop_attributes conv_attributes;
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
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(ConvolutionFwdNodeTests, PreValidateNodeMissingValues)
{
    Convolution_fprop_attributes conv_attributes;

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
    Convolution_fprop_attributes conv_attributes2;
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
    Convolution_fprop_attributes conv_attributes3;
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
    Convolution_fprop_attributes conv_attributes4;
    conv_attributes4.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes4.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes4.set_y(std::make_shared<Tensor_attributes>());
    ConvolutionNode node_without_params(std::move(conv_attributes4), graph_attributes);

    error = node_without_params.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    // Test with all values
    Convolution_fprop_attributes conv_attributes5;
    conv_attributes5.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes5.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes5.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes5.set_pre_padding({1, 1});
    conv_attributes5.set_post_padding({1, 1});
    conv_attributes5.set_stride({1, 1});
    conv_attributes5.set_dilation({1, 1});
    ConvolutionNode node_with_all_values(std::move(conv_attributes5), graph_attributes);

    error = node_with_all_values.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNode)
{
    Convolution_fprop_attributes conv_attributes;
    conv_attributes.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes.set_pre_padding({1, 1});
    conv_attributes.set_post_padding({1, 1});
    conv_attributes.set_stride({1, 1});
    conv_attributes.set_dilation({1, 1});

    auto input_tensor = conv_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 3, 32, 32}) // NCHW format
        .set_stride({3072, 1024, 32, 1});

    auto weights_tensor = conv_attributes.get_w();
    weights_tensor->set_uid(2)
        .set_name("WeightsTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({64, 3, 3, 3}) // KCHW format
        .set_stride({27, 9, 3, 1});

    auto output_tensor = conv_attributes.get_y();
    output_tensor->set_uid(3).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    // Expected output dimensions: (32 + 1 + 1 - 3) / 1 + 1 = 32
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 64, 32, 32}));
    EXPECT_FALSE(output_tensor->get_stride().empty());
}

TEST(ConvolutionFwdNodeTests, InferPropertiesNodeWithStrideAndPadding)
{
    Convolution_fprop_attributes conv_attributes;
    conv_attributes.set_x(std::make_shared<Tensor_attributes>());
    conv_attributes.set_w(std::make_shared<Tensor_attributes>());
    conv_attributes.set_y(std::make_shared<Tensor_attributes>());
    conv_attributes.set_pre_padding({2, 2});
    conv_attributes.set_post_padding({2, 2});
    conv_attributes.set_stride({2, 2});
    conv_attributes.set_dilation({1, 1});

    auto input_tensor = conv_attributes.get_x();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 3, 32, 32}) // NCHW format
        .set_stride({3072, 1024, 32, 1});

    auto weights_tensor = conv_attributes.get_w();
    weights_tensor->set_uid(2)
        .set_name("WeightsTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({64, 3, 5, 5}) // KCHW format
        .set_stride({75, 25, 5, 1});

    auto output_tensor = conv_attributes.get_y();
    output_tensor->set_uid(3).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    ConvolutionNode node(std::move(conv_attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    // Expected output dimensions: (32 + 2 + 2 - 5) / 2 + 1 = 16
    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 64, 16, 16}));
    EXPECT_FALSE(output_tensor->get_stride().empty());
}

TEST(ConvolutionFwdNodeTests, PackNode)
{
    Convolution_fprop_attributes conv_attributes;
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
    conv_attributes.set_conv_mode(hipdnn_sdk::data_objects::ConvMode_CROSS_CORRELATION);

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
    Convolution_fprop_attributes conv_attributes;
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

    EXPECT_TRUE(used_ids.find(1) != used_ids.end());
    EXPECT_TRUE(used_ids.find(2) != used_ids.end());
    EXPECT_TRUE(used_ids.find(3) != used_ids.end());
}

TEST(ConvolutionFwdNodeTests, PopulateHipdnnTensorIds)
{
    Convolution_fprop_attributes conv_attributes;
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
    EXPECT_EQ(error.code, error_code_t::OK);

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
