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

class TestMiopenBatchnormPlanBuilder : public ::testing::Test
{
protected:
    MiopenBatchnormPlanBuilder planBuilder;
    HipdnnEnginePluginHandle dummyHandle;
};

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    Mock_graph mockGraph;
    EXPECT_CALL(mockGraph, node_count()).WillRepeatedly(::testing::Return(2));

    bool applicable = planBuilder.isApplicable(mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    Mock_graph mockGraph;
    EXPECT_CALL(mockGraph, node_count()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, has_only_supported_attributes(::testing::_))
        .WillOnce(::testing::Return(false));

    bool applicable = planBuilder.isApplicable(mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, IsApplicableReturnsTrueForSupportedSingleNodeGraph)
{
    Mock_graph mockGraph;
    EXPECT_CALL(mockGraph, node_count()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, has_only_supported_attributes(::testing::_))
        .WillOnce(::testing::Return(true));

    bool applicable = planBuilder.isApplicable(mockGraph);

    EXPECT_TRUE(applicable);
}

TEST_F(TestMiopenBatchnormPlanBuilder, GetWorkspaceSizeReturnsExpectedValue)
{
    Mock_graph mockGraph;

    size_t workspaceSize = planBuilder.getWorkspaceSize(dummyHandle, mockGraph);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanSetsPlanForSupportedNode)
{
    // Use a real flatbuffer graph with a valid batchnorm node
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(planBuilder.buildPlan(dummyHandle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestMiopenBatchnormPlanBuilder, BuildPlanThrowsForUnsupportedNodeType)
{
    // Create a graph with a node of unsupported type
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node with NONE attributes type
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder, "unsupported", hipdnn_sdk::data_objects::NodeAttributes_NONE, 0);
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType_FLOAT,
                                                      hipdnn_sdk::data_objects::DataType_HALF,
                                                      hipdnn_sdk::data_objects::DataType_BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);

    hipdnn_plugin::Graph_wrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(planBuilder.buildPlan(dummyHandle, graph, ctx),
                 hipdnn_plugin::Hipdnn_plugin_exception);
    EXPECT_FALSE(ctx.hasValidPlan());
}
