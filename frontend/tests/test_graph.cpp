// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>

using namespace hipdnn_frontend::graph;

TEST(GraphTests, SetAndGetAttributes)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_type(DataType_t::FLOAT)
        .set_intermediate_type(DataType_t::HALF)
        .set_io_type(DataType_t::FLOAT);

    EXPECT_EQ(graph.get_name(), "TestGraph");
    EXPECT_EQ(graph.get_compute_type(), DataType_t::FLOAT);
    EXPECT_EQ(graph.get_intermediate_type(), DataType_t::HALF);
    EXPECT_EQ(graph.get_io_type(), DataType_t::FLOAT);
}

TEST(GraphTests, BatchnormInferenceNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<Tensor_attributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    auto mean         = std::make_shared<Tensor_attributes>();
    auto inv_variance = std::make_shared<Tensor_attributes>();
    auto scale        = std::make_shared<Tensor_attributes>();
    auto bias         = std::make_shared<Tensor_attributes>();

    Batchnorm_inference_attributes attributes;
    attributes.name = "BatchnormNode";

    auto y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
}

TEST(GraphTests, PointwiseNodeCreationSingleInput)
{
    Graph graph;

    auto in_0 = std::make_shared<Tensor_attributes>();
    in_0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    Pointwise_attributes attributes;
    attributes.name = "PointwiseNode";
    attributes.set_operation(PointwiseMode_t::RELU);

    auto out_0 = graph.pointwise(in_0, attributes);

    EXPECT_EQ(out_0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out_0->get_is_virtual());
}

TEST(GraphTests, PointwiseNodeCreationTwoInputs)
{
    Graph graph;

    auto in_0 = std::make_shared<Tensor_attributes>();
    auto in_1 = std::make_shared<Tensor_attributes>();

    in_0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);
    in_1->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    Pointwise_attributes attributes;
    attributes.name = "PointwiseNode";
    attributes.set_operation(PointwiseMode_t::RELU);

    auto out_0 = graph.pointwise(in_0, in_1, attributes);

    EXPECT_EQ(out_0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out_0->get_is_virtual());
}

TEST(GraphTests, PointwiseNodeCreationThreeInputs)
{
    Graph graph;

    auto in_0 = std::make_shared<Tensor_attributes>();
    auto in_1 = std::make_shared<Tensor_attributes>();
    auto in_2 = std::make_shared<Tensor_attributes>();

    in_0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);
    in_1->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);
    in_2->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    Pointwise_attributes attributes;
    attributes.name = "PointwiseNode";
    attributes.set_operation(PointwiseMode_t::RELU);

    auto out_0 = graph.pointwise(in_0, in_1, in_2, attributes);

    EXPECT_EQ(out_0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out_0->get_is_virtual());
}