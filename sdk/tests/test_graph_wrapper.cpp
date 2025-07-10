// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <set>

TEST(Graph_wrapperTest, NullBufferIsInvalid)
{
    Graph_wrapper wrapper(nullptr, 0);
    EXPECT_FALSE(wrapper.is_valid());
}

TEST(Graph_wrapperTest, NonGraphBufferIsInvalid)
{
    auto builder = flatbuffer_test_utils::create_valid_engine_details(123);
    auto serialized_graph = builder.Release();

    Graph_wrapper wrapper(serialized_graph.data(), serialized_graph.size());

    EXPECT_FALSE(wrapper.is_valid());
}

TEST(Graph_wrapperTest, ValidGraphReturnsCorrectNodeCountForEmptyGraph)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::create_empty_valid_graph();
    auto serialized_graph = builder.Release();

    Graph_wrapper wrapper(serialized_graph.data(), serialized_graph.size());

    EXPECT_TRUE(wrapper.is_valid());
    EXPECT_EQ(wrapper.node_count(), 0);
}

TEST(Graph_wrapperTest, ValidGraphReturnsCorrectNodeCount)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();

    Graph_wrapper wrapper(serialized_graph.data(), serialized_graph.size());

    EXPECT_TRUE(wrapper.is_valid());
    EXPECT_EQ(wrapper.node_count(), 1);
}

TEST(Graph_wrapperTest, HasSupportedTypesReturnsTrueIfAllSupported)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();

    Graph_wrapper wrapper(serialized_graph.data(), serialized_graph.size());

    std::set<hipdnn_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes};
    EXPECT_TRUE(wrapper.has_supported_types(supported));

    supported.insert(hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes);
    EXPECT_TRUE(wrapper.has_supported_types(supported));
}

TEST(Graph_wrapperTest, HasSupportedTypesReturnsFalseIfAnyUnsupported)
{
    flatbuffers::FlatBufferBuilder builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();

    Graph_wrapper wrapper(serialized_graph.data(), serialized_graph.size());

    std::set<hipdnn_sdk::data_objects::NodeAttributes> supported
        = {hipdnn_sdk::data_objects::NodeAttributes_PointwiseAttributes};
    EXPECT_FALSE(wrapper.has_supported_types(supported));

    supported.insert(hipdnn_sdk::data_objects::NodeAttributes_BatchnormAttributes);
    EXPECT_FALSE(wrapper.has_supported_types(supported));
}
