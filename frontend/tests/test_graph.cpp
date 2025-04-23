// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(GraphTests, SetAndGetAttributes)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    EXPECT_EQ(graph.get_name(), "TestGraph");
    EXPECT_EQ(graph.get_compute_data_type(), DataType_t::FLOAT);
    EXPECT_EQ(graph.get_intermediate_data_type(), DataType_t::HALF);
    EXPECT_EQ(graph.get_io_data_type(), DataType_t::FLOAT);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
}

TEST(GraphTests, BatchnormNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<Tensor_attributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    auto bias  = std::make_shared<Tensor_attributes>();

    Batchnorm_attributes attributes;
    attributes.name = "BatchnormNode";
    attributes.set_epsilon(std::make_shared<Tensor_attributes>());

    auto [y, mean, inv_variance, next_running_mean, next_running_variance]
        = graph.batchnorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    EXPECT_EQ(mean->get_name(), "BatchnormNode::MEAN");
    EXPECT_TRUE(mean->get_is_virtual());

    EXPECT_EQ(inv_variance->get_name(), "BatchnormNode::INV_VARIANCE");
    EXPECT_TRUE(inv_variance->get_is_virtual());

    EXPECT_FALSE(next_running_mean);
    EXPECT_FALSE(next_running_variance);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
}

TEST(GraphTests, BatchnormBackwardNodeCreation)
{
    Graph graph;

    auto dy    = std::make_shared<Tensor_attributes>();
    auto x     = std::make_shared<Tensor_attributes>();
    auto scale = std::make_shared<Tensor_attributes>();

    dy->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    Batchnorm_backward_attributes attributes;
    attributes.name = "BatchnormBackwardNode";

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, attributes);

    EXPECT_EQ(dx->get_name(), "BatchnormBackwardNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    EXPECT_EQ(dscale->get_name(), "BatchnormBackwardNode::DSCALE");
    EXPECT_TRUE(dscale->get_is_virtual());

    EXPECT_EQ(dbias->get_name(), "BatchnormBackwardNode::DBIAS");
    EXPECT_TRUE(dbias->get_is_virtual());

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
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

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
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

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
    attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto out_0 = graph.pointwise(in_0, in_1, in_2, attributes);

    EXPECT_EQ(out_0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out_0->get_is_virtual());

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
}

static void validate_tensor(const Tensor_attributes&                           tensor,
                            const hipdnn_sdk::data_objects::TensorAttributesT& serialized_tensor)
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
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

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

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 6);
    EXPECT_EQ(deserialized_graph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
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
    EXPECT_EQ(
        deserialized_graph->nodes[0]->attributes.type,
        hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormInferenceAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y, y->get_uid());
}

TEST(GraphTests, BuildAndSerializeBatchnormGraph)
{
    Graph graph;

    graph.set_name("SerializedBatchnormGraph")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto bias = std::make_shared<Tensor_attributes>();
    bias->set_uid(3).set_name("Bias").set_data_type(DataType_t::FLOAT);

    auto prev_running_mean = std::make_shared<Tensor_attributes>();
    prev_running_mean->set_uid(4).set_name("PrevRunningMean").set_data_type(DataType_t::FLOAT);

    auto prev_running_variance = std::make_shared<Tensor_attributes>();
    prev_running_variance->set_uid(5)
        .set_name("PrevRunningVariance")
        .set_data_type(DataType_t::FLOAT);

    auto momentum = std::make_shared<Tensor_attributes>();
    momentum->set_uid(6).set_name("Momentum").set_data_type(DataType_t::FLOAT);

    auto epsilon = std::make_shared<Tensor_attributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_data_type(DataType_t::FLOAT);

    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormNode";
    batchnorm_attributes.set_previous_running_stats(
        prev_running_mean, prev_running_variance, momentum);
    batchnorm_attributes.set_epsilon(epsilon);

    auto [y, mean, inv_variance, next_running_mean, next_running_variance]
        = graph.batchnorm(x, scale, bias, batchnorm_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedBatchnormGraph");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 12);
    EXPECT_EQ(deserialized_graph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*x, tensor_lookup[x->get_uid()]);
    validate_tensor(*scale, tensor_lookup[scale->get_uid()]);
    validate_tensor(*bias, tensor_lookup[bias->get_uid()]);
    validate_tensor(*epsilon, tensor_lookup[epsilon->get_uid()]);
    validate_tensor(*prev_running_mean, tensor_lookup[prev_running_mean->get_uid()]);
    validate_tensor(*prev_running_variance, tensor_lookup[prev_running_variance->get_uid()]);
    validate_tensor(*momentum, tensor_lookup[momentum->get_uid()]);
    validate_tensor(*y, tensor_lookup[y->get_uid()]);
    validate_tensor(*mean, tensor_lookup[mean->get_uid()]);
    validate_tensor(*inv_variance, tensor_lookup[inv_variance->get_uid()]);
    validate_tensor(*next_running_mean, tensor_lookup[next_running_mean->get_uid()]);
    validate_tensor(*next_running_variance, tensor_lookup[next_running_variance->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchnormAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->epsilon, epsilon->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_mean, prev_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_variance,
              prev_running_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->momentum, momentum->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y, y->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_mean, next_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_variance,
              next_running_variance->get_uid());
}

TEST(GraphTests, BuildAndSerializeBatchnormAndPointwiseGraph)
{
    Graph graph;

    graph.set_name("SerializedBatchnormAndPointwiseGraph")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto bias = std::make_shared<Tensor_attributes>();
    bias->set_uid(3).set_name("Bias").set_data_type(DataType_t::FLOAT);

    auto prev_running_mean = std::make_shared<Tensor_attributes>();
    prev_running_mean->set_uid(4).set_name("PrevRunningMean").set_data_type(DataType_t::FLOAT);

    auto prev_running_variance = std::make_shared<Tensor_attributes>();
    prev_running_variance->set_uid(5)
        .set_name("PrevRunningVariance")
        .set_data_type(DataType_t::FLOAT);

    auto momentum = std::make_shared<Tensor_attributes>();
    momentum->set_uid(6).set_name("Momentum").set_data_type(DataType_t::FLOAT);

    auto epsilon = std::make_shared<Tensor_attributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_data_type(DataType_t::FLOAT);

    Batchnorm_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormNode";
    batchnorm_attributes.set_previous_running_stats(
        prev_running_mean, prev_running_variance, momentum);
    batchnorm_attributes.set_epsilon(epsilon);

    auto [y, mean, inv_variance, next_running_mean, next_running_variance]
        = graph.batchnorm(x, scale, bias, batchnorm_attributes);

    Pointwise_attributes pointwise_attributes;
    pointwise_attributes.name = "PointwiseNode";
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto out_0 = graph.pointwise(y, pointwise_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedBatchnormAndPointwiseGraph");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 13);
    EXPECT_EQ(deserialized_graph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*x, tensor_lookup[x->get_uid()]);
    validate_tensor(*scale, tensor_lookup[scale->get_uid()]);
    validate_tensor(*bias, tensor_lookup[bias->get_uid()]);
    validate_tensor(*epsilon, tensor_lookup[epsilon->get_uid()]);
    validate_tensor(*prev_running_mean, tensor_lookup[prev_running_mean->get_uid()]);
    validate_tensor(*prev_running_variance, tensor_lookup[prev_running_variance->get_uid()]);
    validate_tensor(*momentum, tensor_lookup[momentum->get_uid()]);
    validate_tensor(*y, tensor_lookup[y->get_uid()]);
    validate_tensor(*mean, tensor_lookup[mean->get_uid()]);
    validate_tensor(*inv_variance, tensor_lookup[inv_variance->get_uid()]);
    validate_tensor(*next_running_mean, tensor_lookup[next_running_mean->get_uid()]);
    validate_tensor(*next_running_variance, tensor_lookup[next_running_variance->get_uid()]);
    validate_tensor(*out_0, tensor_lookup[out_0->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchnormAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->epsilon, epsilon->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_mean, prev_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_variance,
              prev_running_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->momentum, momentum->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y, y->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_mean, next_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_variance,
              next_running_variance->get_uid());

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0, y->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);
}

TEST(GraphTests, BuildAndSerializePointwiseGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    auto in_0 = std::make_shared<Tensor_attributes>();
    in_0->set_uid(1)
        .set_name("Input0")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    Pointwise_attributes pointwise_attributes;
    pointwise_attributes.name = "PointwiseNode";
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto out_0 = graph.pointwise(in_0, pointwise_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 2);
    EXPECT_EQ(deserialized_graph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*in_0, tensor_lookup[in_0->get_uid()]);
    validate_tensor(*out_0, tensor_lookup[out_0->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0, in_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);
}

TEST(GraphTests, BuildAndSerializePointwiseAndBatchnormInferenceGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

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
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto out_0 = graph.pointwise(y, pointwise_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 7);
    EXPECT_EQ(deserialized_graph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
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
    EXPECT_EQ(
        deserialized_graph->nodes[0]->attributes.type,
        hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormInferenceAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y, y->get_uid());

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0, y->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);
}

TEST(GraphTests, BuildAndSerializeBatchnormBackwardGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    auto dy = std::make_shared<Tensor_attributes>();
    dy->set_uid(1)
        .set_name("Dy")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(2)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(3).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto mean = std::make_shared<Tensor_attributes>();
    mean->set_uid(4).set_name("Mean").set_data_type(DataType_t::FLOAT);

    auto inv_variance = std::make_shared<Tensor_attributes>();
    inv_variance->set_uid(5).set_name("InvVariance").set_data_type(DataType_t::FLOAT);

    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormBackwardNode";
    batchnorm_attributes.set_saved_mean_and_inv_variance(mean, inv_variance);

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, batchnorm_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 8);
    EXPECT_EQ(deserialized_graph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*dy, tensor_lookup[dy->get_uid()]);
    validate_tensor(*x, tensor_lookup[x->get_uid()]);
    validate_tensor(*scale, tensor_lookup[scale->get_uid()]);
    validate_tensor(*mean, tensor_lookup[mean->get_uid()]);
    validate_tensor(*inv_variance, tensor_lookup[inv_variance->get_uid()]);
    validate_tensor(*dx, tensor_lookup[dx->get_uid()]);
    validate_tensor(*dscale, tensor_lookup[dscale->get_uid()]);
    validate_tensor(*dbias, tensor_lookup[dbias->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "BatchnormBackwardNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormBackwardAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->dy, dy->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dx, dx->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dscale, dscale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dbias, dbias->get_uid());
}

TEST(GraphTests, BuildAndSerializePointwiseAndBatchnormBackwardGraph)
{
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::HALF)
        .set_io_data_type(DataType_t::FLOAT);

    auto x_pointwise = std::make_shared<Tensor_attributes>();
    x_pointwise->set_uid(6)
        .set_name("X_Pointwise")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    Pointwise_attributes pointwise_attributes;
    pointwise_attributes.name = "PointwiseNode";
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);

    auto dy = graph.pointwise(x_pointwise, pointwise_attributes);

    auto x = std::make_shared<Tensor_attributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType_t::FLOAT);

    auto mean = std::make_shared<Tensor_attributes>();
    mean->set_uid(3).set_name("Mean").set_data_type(DataType_t::FLOAT);

    auto inv_variance = std::make_shared<Tensor_attributes>();
    inv_variance->set_uid(4).set_name("InvVariance").set_data_type(DataType_t::FLOAT);

    Batchnorm_backward_attributes batchnorm_attributes;
    batchnorm_attributes.name = "BatchnormBackwardNode";
    batchnorm_attributes.set_saved_mean_and_inv_variance(mean, inv_variance);

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, batchnorm_attributes);

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();

    auto build_result = graph.build_operation_graph();
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

    auto deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(graph.serialized_graph.data());
    ASSERT_NE(deserialized_graph, nullptr);

    EXPECT_EQ(deserialized_graph->name, "SerializedGraphTest");
    EXPECT_EQ(deserialized_graph->compute_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->intermediate_type, hipdnn_sdk::data_objects::DataType_HALF);
    EXPECT_EQ(deserialized_graph->io_type, hipdnn_sdk::data_objects::DataType_FLOAT);
    EXPECT_EQ(deserialized_graph->tensors.size(), 9);
    EXPECT_EQ(deserialized_graph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensor_lookup;
    for(auto& tensor : deserialized_graph->tensors)
    {
        tensor_lookup[tensor->uid] = *tensor;
    }

    validate_tensor(*x_pointwise, tensor_lookup[x_pointwise->get_uid()]);
    validate_tensor(*x, tensor_lookup[x->get_uid()]);
    validate_tensor(*scale, tensor_lookup[scale->get_uid()]);
    validate_tensor(*mean, tensor_lookup[mean->get_uid()]);
    validate_tensor(*inv_variance, tensor_lookup[inv_variance->get_uid()]);
    validate_tensor(*dy, tensor_lookup[dy->get_uid()]);
    validate_tensor(*dx, tensor_lookup[dx->get_uid()]);
    validate_tensor(*dscale, tensor_lookup[dscale->get_uid()]);
    validate_tensor(*dbias, tensor_lookup[dbias->get_uid()]);

    EXPECT_EQ(deserialized_graph->nodes[0]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0, x_pointwise->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0, dy->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "BatchnormBackwardNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormBackwardAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[1]->attributes.AsBatchnormBackwardAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->dy, dy->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->x, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dx, dx->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dscale, dscale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dbias, dbias->get_uid());
}
// NOLINTEND
