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
    MiopenBatchnormPlanBuilder plan_builder;
    HipdnnEnginePluginHandle dummy_handle;
};

TEST_F(Test_miopen_batchnorm_plan_builder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    bool applicable = plan_builder.isApplicable(mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(Test_miopen_batchnorm_plan_builder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    bool applicable = plan_builder.isApplicable(mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(Test_miopen_batchnorm_plan_builder, IsApplicableReturnsTrueForSupportedSingleNodeGraph)
{
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(true));

    bool applicable = plan_builder.isApplicable(mockGraph);

    EXPECT_TRUE(applicable);
}

TEST_F(Test_miopen_batchnorm_plan_builder, GetWorkspaceSizeReturnsExpectedValue)
{
    MockGraph mockGraph;

    size_t workspace_size = plan_builder.getWorkspaceSize(dummy_handle, mockGraph);

    EXPECT_EQ(workspace_size, 0u);
}

TEST_F(Test_miopen_batchnorm_plan_builder, BuildPlanSetsPlanForSupportedNode)
{
    // Use a real flatbuffer graph with a valid batchnorm node
    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    // Should not throw
    EXPECT_NO_THROW(plan_builder.buildPlan(dummy_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
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

    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(plan_builder.buildPlan(dummy_handle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}
