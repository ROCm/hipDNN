// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/graph.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include "fake_backend/fake_hipdnn_backend.hpp"
#include "fake_backend/mock_hipdnn_backend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace ::testing;

class Graph_test_fixture : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mock_backend;
    hipdnnHandle_t _handle;

    void SetUp() override
    {
        _mock_backend = std::make_shared<Mock_hipdnn_backend>();
        _handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);

        fake_hipdnn_backend::set_mock_hipdnn_backend(_mock_backend);
    }
    void TearDown() override {}

    void expect_graph_serialized_to_backend_descriptor(
        std::unique_ptr<hipdnn_sdk::data_objects::GraphT>& deserialized_graph)
    {
        EXPECT_CALL(
            *_mock_backend,
            hipdnnBackendCreateAndDeserializeGraph_ext(::testing::_, ::testing::_, ::testing::_))
            .WillOnce([&deserialized_graph](hipdnnBackendDescriptor_t* descriptor,
                                            const uint8_t* serialized_graph,
                                            size_t graph_byte_size) {
                std::ignore = descriptor;
                deserialized_graph = hipdnn_sdk::data_objects::UnPackGraph(serialized_graph);
                EXPECT_NE(deserialized_graph, nullptr);
                EXPECT_GE(graph_byte_size, 0);
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    static std::shared_ptr<Tensor_attributes> create_basic_batchnorm_graph(Graph& graph)
    {
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

        return graph.batchnorm_inference(x, mean, inv_variance, scale, bias, batchnorm_attributes);
    }
};

TEST_F(Graph_test_fixture, SetAndGetAttributes)
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

TEST_F(Graph_test_fixture, BatchnormNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<Tensor_attributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    auto scale = std::make_shared<Tensor_attributes>();
    auto bias = std::make_shared<Tensor_attributes>();

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

TEST_F(Graph_test_fixture, BatchnormBackwardNodeCreation)
{
    Graph graph;

    auto dy = std::make_shared<Tensor_attributes>();
    auto x = std::make_shared<Tensor_attributes>();
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

TEST_F(Graph_test_fixture, BatchnormInferenceNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<Tensor_attributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType_t::FLOAT);

    auto mean = std::make_shared<Tensor_attributes>();
    auto inv_variance = std::make_shared<Tensor_attributes>();
    auto scale = std::make_shared<Tensor_attributes>();
    auto bias = std::make_shared<Tensor_attributes>();

    Batchnorm_inference_attributes attributes;
    attributes.name = "BatchnormNode";

    auto y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validation_result = graph.validate();
    EXPECT_TRUE(validation_result.is_good()) << validation_result.get_message();
}

TEST_F(Graph_test_fixture, PointwiseNodeCreationSingleInput)
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

TEST_F(Graph_test_fixture, PointwiseNodeCreationTwoInputs)
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

TEST_F(Graph_test_fixture, PointwiseNodeCreationThreeInputs)
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

static void validate_tensor(const Tensor_attributes& tensor,
                            const hipdnn_sdk::data_objects::TensorAttributesT& serialized_tensor)
{
    EXPECT_EQ(tensor.get_name(), serialized_tensor.name);
    EXPECT_EQ(tensor.get_uid(), serialized_tensor.uid);
    EXPECT_EQ(to_sdk_type(tensor.get_data_type()), serialized_tensor.data_type);
    EXPECT_EQ(tensor.get_dim(), serialized_tensor.dims);
    EXPECT_EQ(tensor.get_stride(), serialized_tensor.strides);
}

// NOLINTBEGIN
TEST_F(Graph_test_fixture, BuildAndSerializeBatchnormInferenceGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_batchnorm_attributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance_tensor_uid, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y_tensor_uid, y->get_uid());
}

TEST_F(Graph_test_fixture, BuildAndSerializeBatchnormGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_batchnorm_attributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->epsilon_tensor_uid, epsilon->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_mean_tensor_uid,
              prev_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_variance_tensor_uid,
              prev_running_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->momentum_tensor_uid, momentum->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y_tensor_uid, y->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance_tensor_uid, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_mean_tensor_uid,
              next_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_variance_tensor_uid,
              next_running_variance->get_uid());
}

TEST_F(Graph_test_fixture, BuildAndSerializeBatchnormAndPointwiseGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_batchnorm_attributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->epsilon_tensor_uid, epsilon->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_mean_tensor_uid,
              prev_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->prev_running_variance_tensor_uid,
              prev_running_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->momentum_tensor_uid, momentum->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y_tensor_uid, y->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance_tensor_uid, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_mean_tensor_uid,
              next_running_mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->next_running_variance_tensor_uid,
              next_running_variance->get_uid());

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0_tensor_uid, y->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0_tensor_uid, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);
}

TEST_F(Graph_test_fixture, BuildAndSerializePointwiseGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_pointwise_attributes->in_0_tensor_uid, in_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0_tensor_uid, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);
}

TEST_F(Graph_test_fixture, BuildAndSerializePointwiseAndBatchnormInferenceGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_batchnorm_attributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance_tensor_uid, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->y_tensor_uid, y->get_uid());

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes);
    auto deserialized_pointwise_attributes
        = deserialized_graph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserialized_pointwise_attributes->in_0_tensor_uid, y->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0_tensor_uid, out_0->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);
}

TEST_F(Graph_test_fixture, BuildAndSerializeBatchnormBackwardGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_batchnorm_attributes->dy_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance_tensor_uid, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dx_tensor_uid, dx->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dscale_tensor_uid, dscale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dbias_tensor_uid, dbias->get_uid());
}

TEST_F(Graph_test_fixture, BuildAndSerializePointwiseAndBatchnormBackwardGraph)
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

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserialized_graph;
    expect_graph_serialized_to_backend_descriptor(deserialized_graph);

    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good()) << build_result.get_message();

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
    EXPECT_EQ(deserialized_pointwise_attributes->in_0_tensor_uid, x_pointwise->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->out_0_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserialized_pointwise_attributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode_RELU_FWD);

    EXPECT_EQ(deserialized_graph->nodes[1]->name, "BatchnormBackwardNode");
    EXPECT_EQ(deserialized_graph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_BatchnormBackwardAttributes);
    auto deserialized_batchnorm_attributes
        = deserialized_graph->nodes[1]->attributes.AsBatchnormBackwardAttributes();
    EXPECT_EQ(deserialized_batchnorm_attributes->dy_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->inv_variance_tensor_uid, inv_variance->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dx_tensor_uid, dx->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dscale_tensor_uid, dscale->get_uid());
    EXPECT_EQ(deserialized_batchnorm_attributes->dbias_tensor_uid, dbias->get_uid());
}

// Test graph.tensor()
TEST_F(Graph_test_fixture, TensorGraphAttributes)
{
    auto tensor = Graph::tensor(Tensor_attributes()
                                    .set_name("TestTensor")
                                    .set_uid(100)
                                    .set_stride({5, 6, 7, 8})
                                    .set_data_type(DataType_t::FLOAT)
                                    .set_is_virtual(false)
                                    .set_dim({1, 2, 3, 4}));

    EXPECT_EQ(tensor->get_data_type(), DataType_t::FLOAT);
    EXPECT_FALSE(tensor->get_is_virtual());
    EXPECT_EQ(tensor->get_dim(), std::vector<int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(tensor->get_stride(), std::vector<int64_t>({5, 6, 7, 8}));
    EXPECT_EQ(tensor->get_name(), "TestTensor");
    EXPECT_EQ(tensor->get_uid(), 100);
}

// Test graph.tensor_like()
TEST_F(Graph_test_fixture, TensorLikeGraphAttributes)
{
    auto tensor = Graph::tensor(Tensor_attributes()
                                    .set_name("TestTensor")
                                    .set_uid(100)
                                    .set_dim({1, 2, 3, 4})
                                    .set_stride({5, 6, 7, 8})
                                    .set_is_virtual(false)
                                    .set_data_type(DataType_t::FLOAT));

    auto tensor_like = Graph::tensor_like(tensor, "TensorLike");

    EXPECT_EQ(tensor_like->get_data_type(), DataType_t::FLOAT);
    EXPECT_FALSE(tensor_like->get_is_virtual());
    EXPECT_EQ(tensor_like->get_dim(), std::vector<int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(tensor_like->get_stride(), std::vector<int64_t>({5, 6, 7, 8}));
    EXPECT_EQ(tensor_like->get_name(), "TensorLike");
    EXPECT_NE(tensor_like->get_uid(), 100);

    EXPECT_NE(tensor_like, tensor);

    auto tensor_like_noname = Graph::tensor_like(tensor_like);
    EXPECT_EQ(tensor_like_noname->get_name(), "");

    EXPECT_EQ(tensor->get_name(), "TestTensor");
    EXPECT_EQ(tensor->get_uid(), 100);
    EXPECT_NE(tensor->get_uid(), tensor_like_noname->get_uid());
}

TEST_F(Graph_test_fixture, WillCorrectlyBuildOperationGraphDescriptor)
{
    Graph graph;
    auto tensor_attributes = create_basic_batchnorm_graph(graph);

    hipdnnBackendDescriptor_t graph_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mock_backend, hipdnnBackendCreateAndDeserializeGraph_ext(_, _, _))
        .WillOnce([&graph_desc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graph_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(
                    graph_desc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce([this](hipdnnBackendDescriptor_t,
                         hipdnnBackendAttributeName_t,
                         hipdnnBackendAttributeType_t,
                         int64_t,
                         const void* array_of_elements) {
            hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(array_of_elements);
            EXPECT_EQ(handle, this->_handle);
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result.is_good());
}

TEST_F(Graph_test_fixture, CreatingExecutionPlansFailsWithNoGraph)
{
    Graph graph;

    auto result = graph.create_execution_plans(_handle, {HeurMode_t::FALLBACK});
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(),
              "Graph has not been built, build the operation graph first. Cannot create "
              "execution plan.");
}

TEST_F(Graph_test_fixture, CanSuccessfullyCreateExecutionPlans)
{
    Graph graph;
    const std::vector<HeurMode_t> heurModes = {HeurMode_t::FALLBACK};
    std::vector<hipdnnBackendHeurMode_t> backend_modes;
    for(const auto& mode : heurModes)
    {
        backend_modes.push_back(to_backend_type(mode));
    }
    auto tensor_attributes = create_basic_batchnorm_graph(graph);

    hipdnnBackendDescriptor_t graph_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mock_backend, hipdnnBackendCreateAndDeserializeGraph_ext(_, _, _))
        .WillOnce([&graph_desc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graph_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    graph.build_operation_graph(_handle);

    hipdnnBackendDescriptor_t heur_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heur_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heur_desc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(heur_desc,
                                          HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          _))
        .WillOnce([&graph_desc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                const void* array_of_elements) {
            EXPECT_EQ(graph_desc,
                      *static_cast<const hipdnnBackendDescriptor_t*>(array_of_elements));
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(
                    heur_desc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce([&backend_modes](hipdnnBackendDescriptor_t,
                                   hipdnnBackendAttributeName_t,
                                   hipdnnBackendAttributeType_t,
                                   int64_t count,
                                   const void* array_of_elements) {
            EXPECT_EQ(count, static_cast<int64_t>(backend_modes.size()));
            auto modes_ptr = static_cast<const hipdnnBackendHeurMode_t*>(array_of_elements);
            for(size_t i = 0; i < backend_modes.size(); ++i)
            {
                EXPECT_EQ(modes_ptr[i], backend_modes[i]);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(heur_desc));

    // Set up the mock to handle multiple calls with different arguments using .WillOnce()/.WillRepeatedly()
    // First call: element_count query
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendGetAttribute(heur_desc,
                                          HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          0,
                                          _,
                                          nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* element_count,
                     void*) {
            *element_count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    hipdnnBackendDescriptor_t engine_config_desc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engine_config_desc](hipdnnBackendDescriptorType_t,
                                        hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engine_config_desc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendGetAttribute(heur_desc,
                                          HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          _,
                                          NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrieved_count,
                     void*) {
            *retrieved_count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    hipdnnBackendDescriptor_t execution_plan_desc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&execution_plan_desc](hipdnnBackendDescriptorType_t,
                                         hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = execution_plan_desc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mock_backend,
        hipdnnBackendSetAttribute(
            execution_plan_desc, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce([this](hipdnnBackendDescriptor_t,
                         hipdnnBackendAttributeName_t,
                         hipdnnBackendAttributeType_t,
                         int64_t,
                         const void* array_of_elements) {
            hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(array_of_elements);
            EXPECT_EQ(handle, this->_handle);
            return HIPDNN_STATUS_SUCCESS;
        });

    auto exec_plan_result = graph.create_execution_plans(_handle, heurModes);
    EXPECT_TRUE(exec_plan_result.is_good());
}

TEST_F(Graph_test_fixture, CheckSupportFailsIfNoExecutionPlanCreated)
{
    Graph graph;
    auto tensor_attributes = create_basic_batchnorm_graph(graph);

    graph.build_operation_graph(_handle);

    auto result = graph.check_support();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(), "Execution plan descriptor is not created or invalid.");
}

TEST_F(Graph_test_fixture, CheckSupportSucceedsWhenExecutionPlanCreated)
{
    Graph graph;
    const std::vector<HeurMode_t> heur_modes = {HeurMode_t::FALLBACK};
    auto tensor_attributes = create_basic_batchnorm_graph(graph);
    graph.build_operation_graph(_handle);

    ON_CALL(*_mock_backend, hipdnnBackendCreateDescriptor(_, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    ON_CALL(*_mock_backend, hipdnnBackendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mock_backend, hipdnnBackendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* element_count,
                           void*) {
            *element_count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    graph.create_execution_plans(_handle, heur_modes);

    auto result = graph.check_support();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(Graph_test_fixture, EngineConfigAndExecutionPlanAreFinalizedAfterBuildPlans)
{
    Graph graph;
    const std::vector<HeurMode_t> heur_modes = {HeurMode_t::FALLBACK};
    auto tensor_attributes = create_basic_batchnorm_graph(graph);

    ON_CALL(*_mock_backend, hipdnnBackendCreateAndDeserializeGraph_ext(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, hipdnnBackendCreateDescriptor(_, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result.is_good());

    hipdnnBackendDescriptor_t engine_config_desc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engine_config_desc](hipdnnBackendDescriptorType_t,
                                        hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engine_config_desc;
            return HIPDNN_STATUS_SUCCESS;
        });

    hipdnnBackendDescriptor_t execution_plan_desc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&execution_plan_desc](hipdnnBackendDescriptorType_t,
                                         hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = execution_plan_desc;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mock_backend, hipdnnBackendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mock_backend, hipdnnBackendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* element_count,
                           void*) {
            *element_count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    result = graph.create_execution_plans(_handle, heur_modes);
    EXPECT_TRUE(result.is_good());

    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(engine_config_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(execution_plan_desc,
                                          HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          _))
        .WillOnce([&engine_config_desc](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t,
                                        const void* array_of_elements) {
            EXPECT_EQ(engine_config_desc,
                      *static_cast<const hipdnnBackendDescriptor_t*>(array_of_elements));
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(execution_plan_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    result = graph.build_plans();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(Graph_test_fixture, WorkspaceSizeIsRetrievedFromExecutionPlan)
{
    Graph graph;
    const std::vector<HeurMode_t> heur_modes = {HeurMode_t::FALLBACK};
    auto tensor_attributes = create_basic_batchnorm_graph(graph);
    graph.build_operation_graph(_handle);

    EXPECT_CALL(*_mock_backend, hipdnnBackendCreateDescriptor(_, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    hipdnnBackendDescriptor_t execution_plan_desc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&execution_plan_desc](hipdnnBackendDescriptorType_t,
                                         hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = execution_plan_desc;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mock_backend, hipdnnBackendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mock_backend, hipdnnBackendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* element_count,
                           void*) {
            *element_count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    graph.create_execution_plans(_handle, heur_modes);

    int64_t workspace_size = 123454;
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendGetAttribute(execution_plan_desc,
                                          HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                          HIPDNN_TYPE_INT64,
                                          1,
                                          nullptr,
                                          _))
        .WillOnce([workspace_size](hipdnnBackendDescriptor_t,
                                   hipdnnBackendAttributeName_t,
                                   hipdnnBackendAttributeType_t,
                                   int64_t,
                                   int64_t*,
                                   void* array_of_elements) {
            *static_cast<int64_t*>(array_of_elements) = workspace_size;
            return HIPDNN_STATUS_SUCCESS;
        });

    int64_t workspace_size_result = 0;
    auto result = graph.get_workspace_size(workspace_size_result);

    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(workspace_size_result, workspace_size);
}

TEST_F(Graph_test_fixture, ExecutePacksVariantPackAndPassesTheCorrectArguments)
{
    using ::testing::_;
    using ::testing::Invoke;
    using ::testing::NotNull;
    using ::testing::Return;

    Graph graph;
    auto tensor = std::make_shared<Tensor_attributes>();
    tensor->set_uid(42)
        .set_name("InputTensor")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType_t::FLOAT);

    Pointwise_attributes pointwise_attributes;
    pointwise_attributes.name = "PointwiseNode";
    pointwise_attributes.set_mode(PointwiseMode_t::RELU_FWD);
    auto out_tensor = graph.pointwise(tensor, pointwise_attributes);

    // build_operation_graph mocks
    hipdnnBackendDescriptor_t graph_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1000);
    EXPECT_CALL(*_mock_backend, hipdnnBackendCreateAndDeserializeGraph_ext(_, _, _))
        .WillOnce([&graph_desc](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = graph_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(
                    graph_desc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(graph_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // create_execution_plans mocks
    hipdnnBackendDescriptor_t heur_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2000);
    hipdnnBackendDescriptor_t engine_cfg_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3000);
    hipdnnBackendDescriptor_t exec_plan_desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4000);

    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce([&heur_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = heur_desc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(heur_desc,
                                          HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(
                    heur_desc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(heur_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendGetAttribute(heur_desc,
                                          HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          0,
                                          _,
                                          nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* count,
                     void*) {
            *count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce(
            [&engine_cfg_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = engine_cfg_desc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendGetAttribute(heur_desc,
                                          HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          _,
                                          NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* count,
                     void*) {
            *count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce(
            [&exec_plan_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = exec_plan_desc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(
                    exec_plan_desc, HIPDNN_ATTR_EXECUTION_PLAN_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // build_plans mocks
    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(engine_cfg_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(exec_plan_desc,
                                          HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(exec_plan_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // get_workspace_size mock
    int64_t expected_workspace_size = 12345;
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendGetAttribute(exec_plan_desc,
                                          HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                          HIPDNN_TYPE_INT64,
                                          1,
                                          nullptr,
                                          _))
        .WillOnce([expected_workspace_size](hipdnnBackendDescriptor_t,
                                            hipdnnBackendAttributeName_t,
                                            hipdnnBackendAttributeType_t,
                                            int64_t,
                                            int64_t*,
                                            void* ptr) {
            *reinterpret_cast<int64_t*>(ptr) = expected_workspace_size;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Prepare variant pack and workspace
    std::unordered_map<int64_t, void*> variant_pack;
    variant_pack[42] = reinterpret_cast<void*>(0xDEADBEEF);
    variant_pack[22] = reinterpret_cast<void*>(0xBEEFBEEF);
    variant_pack[33] = reinterpret_cast<void*>(0xBEEFDEAD);
    variant_pack[1] = reinterpret_cast<void*>(0xDEADBEE);

    void* workspace = reinterpret_cast<void*>(0xCAFEBABE);

    hipdnnBackendDescriptor_t variant_pack_desc
        = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5000);
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .WillOnce(
            [&variant_pack_desc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = variant_pack_desc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(variant_pack_desc,
                                          HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                          HIPDNN_TYPE_VOID_PTR,
                                          static_cast<int64_t>(variant_pack.size()),
                                          NotNull()))
        .WillOnce(Invoke([variant_pack](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t count,
                                        const void* ptr) {
            EXPECT_EQ(count, 4);
            auto data_ptrs = static_cast<void* const*>(ptr);
            int i = 0;
            for(const auto& value : std::views::values(variant_pack))
            {
                EXPECT_EQ(data_ptrs[i++], value);
            }

            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(variant_pack_desc,
                                          HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                          HIPDNN_TYPE_INT64,
                                          static_cast<int64_t>(variant_pack.size()),
                                          NotNull()))
        .WillOnce(Invoke([variant_pack](hipdnnBackendDescriptor_t,
                                        hipdnnBackendAttributeName_t,
                                        hipdnnBackendAttributeType_t,
                                        int64_t count,
                                        const void* ptr) {
            EXPECT_EQ(count, 4);
            auto keys = static_cast<const int64_t*>(ptr);
            int i = 0;
            for(const auto& key : std::views::keys(variant_pack))
            {
                EXPECT_EQ(keys[i++], key);
            }
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mock_backend,
                hipdnnBackendSetAttribute(variant_pack_desc,
                                          HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                          HIPDNN_TYPE_VOID_PTR,
                                          1,
                                          NotNull()))
        .WillOnce(Invoke([workspace](hipdnnBackendDescriptor_t,
                                     hipdnnBackendAttributeName_t,
                                     hipdnnBackendAttributeType_t,
                                     int64_t count,
                                     const void* ptr) {
            EXPECT_EQ(count, 1);
            auto workspace_ptr = *static_cast<void* const*>(ptr);
            EXPECT_EQ(workspace_ptr, workspace);
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mock_backend, hipdnnBackendFinalize(variant_pack_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mock_backend, hipdnnBackendExecute(_handle, exec_plan_desc, variant_pack_desc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Run the full sequence
    auto build_result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(build_result.is_good());

    std::vector<HeurMode_t> heur_modes = {HeurMode_t::FALLBACK};
    auto plan_result = graph.create_execution_plans(_handle, heur_modes);
    EXPECT_TRUE(plan_result.is_good());

    auto support_result = graph.check_support();
    EXPECT_TRUE(support_result.is_good());

    auto build_plans_result = graph.build_plans();
    EXPECT_TRUE(build_plans_result.is_good());

    int64_t workspace_size = 0;
    auto ws_result = graph.get_workspace_size(workspace_size);
    EXPECT_TRUE(ws_result.is_good());
    EXPECT_EQ(workspace_size, expected_workspace_size);

    auto exec_result = graph.execute(_handle, variant_pack, workspace);
    EXPECT_TRUE(exec_result.is_good());
}

// NOLINTEND
