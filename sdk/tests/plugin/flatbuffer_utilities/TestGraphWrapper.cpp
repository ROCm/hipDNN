// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <set>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_plugin;
using namespace hipdnn_sdk::data_objects;

TEST(GraphWrapperTest, NullBufferIsInvalid)
{
    GraphWrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.isValid());
    EXPECT_THROW(wrapper.getGraph(), HipdnnPluginException);
}

TEST(GraphWrapperTest, NonGraphBufferIsInvalid)
{
    auto builder = hipdnn_backend::test_utilities::createValidEngineDetails(123);
    auto serializedGraph = builder.Release();

    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    EXPECT_FALSE(wrapper.isValid());
}

TEST(GraphWrapperTest, ValidGraphReturnsCorrectNodeCountForEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_backend::test_utilities::createEmptyValidGraph();
    auto serializedGraph = builder.Release();

    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 0);
}

TEST(GraphWrapperTest, ValidGraphReturnsCorrectNodeCount)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_backend::test_utilities::createValidBatchnormGraph();
    auto serializedGraph = builder.Release();

    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.nodeCount(), 1);
}

TEST(GraphWrapperTest, HasSupportedTypesReturnsTrueIfAllSupported)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_backend::test_utilities::createValidBatchnormGraph();
    auto serializedGraph = builder.Release();

    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    std::set<hipdnn_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes};
    EXPECT_TRUE(wrapper.hasOnlySupportedAttributes(supported));

    supported.insert(hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes);
    EXPECT_TRUE(wrapper.hasOnlySupportedAttributes(supported));
}

TEST(GraphWrapperTest, HasSupportedTypesReturnsFalseIfAnyUnsupported)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_backend::test_utilities::createValidBatchnormGraph();
    auto serializedGraph = builder.Release();

    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());

    std::set<hipdnn_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes};
    EXPECT_FALSE(wrapper.hasOnlySupportedAttributes(supported));

    supported.insert(hipdnn_sdk::data_objects::NodeAttributes_BatchnormAttributes);
    EXPECT_FALSE(wrapper.hasOnlySupportedAttributes(supported));
}

TEST(GraphWrapperTest, GetTensorMapEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder
        = hipdnn_backend::test_utilities::createEmptyValidGraph();
    auto serializedGraph = builder.Release();

    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto& tensorMap = wrapper.getTensorMap();
    EXPECT_TRUE(tensorMap.empty());
}

TEST(GraphWrapperTest, GetTensorMapReturnsCorrectTensors)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_sdk::data_objects::DataType_FLOAT, &strides, &dims));

    auto graph = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                             "test",
                                                             DataType_FLOAT,
                                                             DataType_HALF,
                                                             DataType_BFLOAT16,
                                                             &tensorAttributes,
                                                             &nodes);
    builder.Finish(graph);

    auto serializedGraph = builder.Release();
    GraphWrapper wrapper(serializedGraph.data(), serializedGraph.size());
    ASSERT_TRUE(wrapper.isValid());

    const auto& tensorMap = wrapper.getTensorMap();
    EXPECT_EQ(tensorMap.size(), 2);
    EXPECT_NE(tensorMap.find(1), tensorMap.end());
    EXPECT_NE(tensorMap.find(2), tensorMap.end());
    EXPECT_EQ(tensorMap.at(1)->uid(), 1);
    EXPECT_EQ(tensorMap.at(2)->uid(), 2);
}
