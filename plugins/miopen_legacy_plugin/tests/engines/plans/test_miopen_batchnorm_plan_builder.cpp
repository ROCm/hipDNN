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

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

class Test_miopen_batchnorm_solver : public ::testing::Test
{
protected:
    Miopen_batchnorm_plan_builder solver;
    hipdnnEnginePluginHandle dummy_handle;
};

TEST_F(Test_miopen_batchnorm_solver, IsApplicableReturnsFalseForMultiNodeGraph)
{
    Mock_graph mock_graph;
    EXPECT_CALL(mock_graph, node_count()).WillRepeatedly(::testing::Return(2));

    bool applicable = solver.is_applicable(mock_graph);

    EXPECT_FALSE(applicable);
}

TEST_F(Test_miopen_batchnorm_solver, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    Mock_graph mock_graph;
    EXPECT_CALL(mock_graph, node_count()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mock_graph, has_only_supported_attributes(::testing::_))
        .WillOnce(::testing::Return(false));

    bool applicable = solver.is_applicable(mock_graph);

    EXPECT_FALSE(applicable);
}

TEST_F(Test_miopen_batchnorm_solver, IsApplicableReturnsTrueForSupportedSingleNodeGraph)
{
    Mock_graph mock_graph;
    EXPECT_CALL(mock_graph, node_count()).WillOnce(::testing::Return(1));
    EXPECT_CALL(mock_graph, has_only_supported_attributes(::testing::_))
        .WillOnce(::testing::Return(true));

    bool applicable = solver.is_applicable(mock_graph);

    EXPECT_TRUE(applicable);
}

TEST_F(Test_miopen_batchnorm_solver, GetWorkspaceSizeReturnsExpectedValue)
{
    Mock_graph mock_graph;

    size_t workspace_size = solver.get_workspace_size(dummy_handle, mock_graph);

    EXPECT_EQ(workspace_size, 0u);
}
