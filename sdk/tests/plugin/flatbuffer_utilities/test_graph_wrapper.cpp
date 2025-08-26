// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <set>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

using namespace hipdnn_plugin;
using namespace hipdnn_sdk::data_objects;

TEST(Graph_wrapperTest, NullBufferIsInvalid)
{
    GraphWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.getGraph(), HipdnnPluginException);
}

TEST(Graph_wrapperTest, NonGraphBufferIsInvalid)
{
    auto builder = flatbuffer_test_utils::createValidEngineDetails(123);
    auto serialized_graph = builder.Release();

    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());

    EXPECT_FALSE(wrapper.isValid());
}

TEST(Graph_wrapperTest, ValidGraphReturnsCorrectNodeCountForEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::createEmptyValidGraph();
    auto serialized_graph = builder.Release();

    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 0);
}

TEST(Graph_wrapperTest, ValidGraphReturnsCorrectNodeCount)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::createValidBatchnormGraph();
    auto serialized_graph = builder.Release();

    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 1);
}

TEST(Graph_wrapperTest, HasSupportedTypesReturnsTrueIfAllSupported)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::createValidBatchnormGraph();
    auto serialized_graph = builder.Release();

    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());

    std::set<hipdnn_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes};
    EXPECT_TRUE(wrapper.hasOnlySupportedAttributes(supported));

    supported.insert(hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes);
    EXPECT_TRUE(wrapper.hasOnlySupportedAttributes(supported));
}

TEST(Graph_wrapperTest, HasSupportedTypesReturnsFalseIfAnyUnsupported)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::createValidBatchnormGraph();
    auto serialized_graph = builder.Release();

    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());

    std::set<hipdnn_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes};
    EXPECT_FALSE(wrapper.hasOnlySupportedAttributes(supported));

    supported.insert(hipdnn_sdk::data_objects::NodeAttributes_BatchnormAttributes);
    EXPECT_FALSE(wrapper.hasOnlySupportedAttributes(supported));
}

TEST(Graph_wrapperTest, GetTensorMapEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::createEmptyValidGraph();
    auto serialized_graph = builder.Release();

    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto& tensor_map = wrapper.getTensorMap();
    EXPECT_TRUE(tensor_map.empty());
}

TEST(Graph_wrapperTest, GetTensorMapReturnsCorrectTensors)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;
    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    tensor_attributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

    auto graph = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                             "test",
                                                             DataType_FLOAT,
                                                             DataType_HALF,
                                                             DataType_BFLOAT16,
                                                             &tensor_attributes,
                                                             &nodes);
    builder.Finish(graph);

    auto serialized_graph = builder.Release();
    GraphWrapper wrapper(serialized_graph.data(), serialized_graph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto& tensor_map = wrapper.getTensorMap();
    EXPECT_EQ(tensor_map.size(), 2);
    EXPECT_NE(tensor_map.find(1), tensor_map.end());
    EXPECT_NE(tensor_map.find(2), tensor_map.end());
    EXPECT_EQ(tensor_map.at(1)->uid(), 1);
    EXPECT_EQ(tensor_map.at(2)->uid(), 2);
}
