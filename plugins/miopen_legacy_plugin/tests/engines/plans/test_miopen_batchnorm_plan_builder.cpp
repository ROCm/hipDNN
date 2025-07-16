/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/test_utils/mock_graph.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

#include "engines/plans/miopen_batchnorm_plan_builder.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

#include "mocks/mock_hipdnn_engine_plugin_execution_context.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

class Test_miopen_batchnorm_plan_builder : public ::testing::Test
{
protected:
    Miopen_batchnorm_plan_builder plan_builder;
    hipdnnEnginePluginHandle dummy_handle;
};

TEST_F(Test_miopen_batchnorm_plan_builder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    Mock_graph mock_graph;
    EXPECT_CALL(mock_graph, node_count()).WillRepeatedly(::testing::Return(2));

    bool applicable = plan_builder.is_applicable(mock_graph);

    EXPECT_FALSE(applicable);
}

TEST_F(Test_miopen_batchnorm_plan_builder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    Mock_graph mock_graph;
    EXPECT_CALL(mock_graph, node_count()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mock_graph, has_only_supported_attributes(::testing::_))
        .WillOnce(::testing::Return(false));

    bool applicable = plan_builder.is_applicable(mock_graph);

    EXPECT_FALSE(applicable);
}

TEST_F(Test_miopen_batchnorm_plan_builder, IsApplicableReturnsTrueForSupportedSingleNodeGraph)
{
    Mock_graph mock_graph;
    EXPECT_CALL(mock_graph, node_count()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mock_graph, has_only_supported_attributes(::testing::_))
        .WillOnce(::testing::Return(true));

    bool applicable = plan_builder.is_applicable(mock_graph);

    EXPECT_TRUE(applicable);
}

TEST_F(Test_miopen_batchnorm_plan_builder, GetWorkspaceSizeReturnsExpectedValue)
{
    Mock_graph mock_graph;

    size_t workspace_size = plan_builder.get_workspace_size(dummy_handle, mock_graph);

    EXPECT_EQ(workspace_size, 0u);
}

TEST_F(Test_miopen_batchnorm_plan_builder, BuildPlanSetsPlanForSupportedNode)
{
    // Use a real flatbuffer graph with a valid batchnorm node
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());
    hipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(plan_builder.build_plan(dummy_handle, graph, ctx));
    EXPECT_TRUE(ctx.has_valid_plan());
}

TEST_F(Test_miopen_batchnorm_plan_builder, BuildPlanThrowsForUnsupportedNodeType)
{
    // Create a graph with a node of unsupported type
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
        tensor_attributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node with NONE attributes type
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "unsupported", hipdnn_sdk::data_objects::NodeAttributes_NONE, 0);
    nodes.push_back(node);

    auto graph_offset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_HALF,
                                                      hipdnn_sdk::data_objects::DataType_BFLOAT16,
                                                      &tensor_attributes,
                                                      &nodes);
    builder.Finish(graph_offset);

    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());

    hipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(plan_builder.build_plan(dummy_handle, graph, ctx),
                 hipdnn_plugin::Hipdnn_plugin_exception);
    EXPECT_FALSE(ctx.has_valid_plan());
}
