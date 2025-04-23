// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/graph_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/attributes/tensor_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/node/pointwise_node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(PointwiseNodeTests, SingleInput)
{
    Pointwise_attributes attributes;
    attributes.set_input_0(std::make_shared<Tensor_attributes>());
    attributes.set_output_0(std::make_shared<Tensor_attributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto input_tensor = attributes.get_input_0();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = attributes.get_output_0();
    output_tensor->set_uid(2).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, TwoInputs)
{
    Pointwise_attributes attributes;
    attributes.set_input_0(std::make_shared<Tensor_attributes>());
    attributes.set_input_1(std::make_shared<Tensor_attributes>());
    attributes.set_output_0(std::make_shared<Tensor_attributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto input_tensor0 = attributes.get_input_0();
    input_tensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto input_tensor1 = attributes.get_input_1();
    input_tensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = attributes.get_output_0();
    output_tensor->set_uid(3).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, ThreeInputs)
{
    Pointwise_attributes attributes;
    attributes.set_input_0(std::make_shared<Tensor_attributes>());
    attributes.set_input_1(std::make_shared<Tensor_attributes>());
    attributes.set_input_2(std::make_shared<Tensor_attributes>());
    attributes.set_output_0(std::make_shared<Tensor_attributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto input_tensor0 = attributes.get_input_0();
    input_tensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto input_tensor1 = attributes.get_input_1();
    input_tensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto input_tensor2 = attributes.get_input_2();
    input_tensor2->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = attributes.get_output_0();
    output_tensor->set_uid(4).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, PreValidateNode)
{
    Pointwise_attributes attributes;
    attributes.set_input_0(std::make_shared<Tensor_attributes>());
    attributes.set_output_0(std::make_shared<Tensor_attributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(PointwiseNodeTests, PreValidateNodeMissingValues)
{
    Pointwise_attributes attributes;

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    attributes.set_input_0(std::make_shared<Tensor_attributes>());
    auto          attributes_copy = attributes;
    PointwiseNode node_with_input(std::move(attributes_copy), graph_attributes);

    error = node_with_input.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    attributes.set_output_0(std::make_shared<Tensor_attributes>());
    attributes_copy = attributes;
    PointwiseNode node_with_output(std::move(attributes_copy), graph_attributes);

    error = node_with_output.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);

    attributes.set_mode(PointwiseMode_t::RELU_FWD);
    attributes_copy = attributes;
    PointwiseNode node_with_all_values(std::move(attributes_copy), graph_attributes);

    error = node_with_all_values.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(PointwiseNodeTests, InferPropertiesNode)
{
    Pointwise_attributes attributes;
    attributes.set_input_0(std::make_shared<Tensor_attributes>());
    attributes.set_output_0(std::make_shared<Tensor_attributes>());
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto input_tensor = attributes.get_input_0();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto output_tensor = attributes.get_output_0();
    output_tensor->set_uid(2).set_name("OutputTensor");

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK);

    EXPECT_EQ(output_tensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(output_tensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(PointwiseNodeTests, PackNode)
{
    Pointwise_attributes attributes;
    attributes.name = "PointwiseNode";

    auto input_tensor = std::make_shared<Tensor_attributes>();
    input_tensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    attributes.set_input_0(input_tensor);

    auto output_tensor = std::make_shared<Tensor_attributes>();
    output_tensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    attributes.set_output_0(output_tensor);

    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    Graph_attributes graph_attributes;
    PointwiseNode    node(std::move(attributes), graph_attributes);

    flatbuffers::FlatBufferBuilder builder;
    auto                           offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto buffer_pointer  = builder.GetBufferPointer();
    auto node_flatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(buffer_pointer);

    EXPECT_STREQ(node_flatbuffer->name()->c_str(), "PointwiseNode");
    EXPECT_EQ(node_flatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes);

    auto packed_attributes = node_flatbuffer->attributes_as_PointwiseAttributes();
    ASSERT_NE(packed_attributes, nullptr);

    EXPECT_EQ(packed_attributes->in_0(), input_tensor->get_uid());
    EXPECT_EQ(packed_attributes->out_0(), output_tensor->get_uid());
    EXPECT_EQ(packed_attributes->operation(), static_cast<int>(PointwiseMode_t::RELU_FWD));
}
