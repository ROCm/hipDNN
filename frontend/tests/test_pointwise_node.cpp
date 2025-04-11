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
    attributes.set_operation(PointwiseMode_t::RELU);

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
    attributes.set_operation(PointwiseMode_t::RELU);

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
    attributes.set_operation(PointwiseMode_t::RELU);

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
