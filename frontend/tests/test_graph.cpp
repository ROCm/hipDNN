// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include "graph_generated.h"

using namespace hipdnn_frontend;
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

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
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

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
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

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
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

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
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

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
}

static void validate_tensor(const Tensor_attributes&              tensor,
                            const hipdnn::sdk::TensorAttributesT& serialized_tensor)
{
    EXPECT_EQ(tensor.get_name(), serialized_tensor.name);
    EXPECT_EQ(tensor.get_uid(), serialized_tensor.uid);
    EXPECT_EQ(to_sdk_type(tensor.get_data_type()), serialized_tensor.data_type);
    EXPECT_EQ(tensor.get_dim(), serialized_tensor.dims);
    EXPECT_EQ(tensor.get_stride(), serialized_tensor.strides);
}

// NOLINTBEGIN
TEST(GraphTests, BuildAndSerializeBatchnormInferenceGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_type(DataType_t::FLOAT)
        .set_intermediate_type(DataType_t::HALF)
        .set_io_type(DataType_t::FLOAT);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto mean = std::make_shared<Tensor_attributes>();
    mean->set_uid(2).set_name("Mean").set_data_type(DataType_t::FLOAT);

    auto inv_variance = std::make_shared<Tensor_attributes>();
    inv_variance->set_uid(3).set_name("InvVariance").set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(4).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto bias = std::make_shared<Tensor_attributes>();
    bias->set_uid(5).set_name("Bias").set_data_type(DataType_t::FLOAT);

    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormNode";

    auto y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, batchnorm_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn::sdk::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn::sdk::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn::sdk::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn::sdk::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 6);
    EXPECT_EQ(deserialized_graph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn::sdk::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*x, tensor_lookup[x->get_uid()]);
    validate_tensor(*mean, tensor_lookup[mean->get_uid()]);
    validate_tensor(*inv_variance, tensor_lookup[inv_variance->get_uid()]);
    validate_tensor(*scale, tensor_lookup[scale->get_uid()]);
    validate_tensor(*bias, tensor_lookup[bias->get_uid()]);
    validate_tensor(*y, tensor_lookup[y->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn::sdk::NodeAttributes::NodeAttributes_BatchNormInferenceAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchNormInferenceAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y, y->get_uid());
}

TEST(GraphTests, BuildAndSerializePointwiseGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_type(DataType_t::FLOAT)
        .set_intermediate_type(DataType_t::HALF)
        .set_io_type(DataType_t::FLOAT);

    auto in_0 = std::make_shared<Tensor_attributes>();
    in_0->set_uid(1)
        .set_name("Input0")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    Pointwise_attributes pointwise_attributes;
    pointwise_attributes.name = "PointwiseNode";
    pointwise_attributes.set_operation(PointwiseMode_t::RELU);

    auto out_0 = graph.pointwise(in_0, pointwise_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn::sdk::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn::sdk::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn::sdk::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn::sdk::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 2);
    EXPECT_EQ(deserialized_graph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn::sdk::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*in_0, tensor_lookup[in_0->get_uid()]);
    validate_tensor(*out_0, tensor_lookup[out_0->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn::sdk::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0, in_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation, hipdnn::sdk::PointwiseMode_RELU_FWD);
}

TEST(GraphTests, BuildAndSerializePointwiseAndBatchnormGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_type(DataType_t::FLOAT)
        .set_intermediate_type(DataType_t::HALF)
        .set_io_type(DataType_t::FLOAT);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto mean = std::make_shared<Tensor_attributes>();
    mean->set_uid(2).set_name("Mean").set_data_type(DataType_t::FLOAT);

    auto inv_variance = std::make_shared<Tensor_attributes>();
    inv_variance->set_uid(3).set_name("InvVariance").set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(4).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto bias = std::make_shared<Tensor_attributes>();
    bias->set_uid(5).set_name("Bias").set_data_type(DataType_t::FLOAT);

    Batchnorm_inference_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormNode";

    auto y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, batchnorm_attributes);

    Pointwise_attributes pointwise_attributes;
    pointwise_attributes.name = "PointwiseNode";
    pointwise_attributes.set_operation(PointwiseMode_t::RELU);

    auto out_0 = graph.pointwise(y, pointwise_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn::sdk::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn::sdk::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn::sdk::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn::sdk::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 7);
    EXPECT_EQ(deserialized_graph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn::sdk::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*x, tensor_lookup[x->get_uid()]);
    validate_tensor(*mean, tensor_lookup[mean->get_uid()]);
    validate_tensor(*inv_variance, tensor_lookup[inv_variance->get_uid()]);
    validate_tensor(*scale, tensor_lookup[scale->get_uid()]);
    validate_tensor(*bias, tensor_lookup[bias->get_uid()]);
    validate_tensor(*y, tensor_lookup[y->get_uid()]);
    validate_tensor(*out_0, tensor_lookup[out_0->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn::sdk::NodeAttributes::NodeAttributes_BatchNormInferenceAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchNormInferenceAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y, y->get_uid());

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn::sdk::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0, y->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation, hipdnn::sdk::PointwiseMode_RELU_FWD);
}
// NOLINTEND
