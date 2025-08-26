// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_details_wrapper.hpp>
#include <hipdnn_sdk/plugin/test_utils/mock_graph.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

#include "engines/miopen_engine.hpp"
#include "mocks/mock_hipdnn_engine_plugin_execution_context.hpp"
#include "mocks/mock_plan_builder.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

TEST(Miopen_engineTest, ConstructorAndId)
{
    MiopenEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    MiopenEngine engine(1);

    MockGraph mock_graph;

    HipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.getWorkspaceSize(dummy_handle, mock_graph), 0u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mock_plan_builder));

    MockGraph mock_graph;

    HipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.getWorkspaceSize(dummy_handle, mock_graph), 1337u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    EXPECT_CALL(*mock_plan_builder, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));
    EXPECT_CALL(*mock_plan_builder2, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder2, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mock_plan_builder));
    engine.addPlanBuilder(std::move(mock_plan_builder2));

    MockGraph mock_graph;

    HipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.getWorkspaceSize(dummy_handle, mock_graph), 45000u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, isApplicable(::testing::_)).WillOnce(::testing::Return(false));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mock_plan_builder));

    MockGraph mock_graph;

    HipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.getWorkspaceSize(dummy_handle, mock_graph), 0u);
}

TEST(Miopen_engineTest, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, isApplicable(::testing::_)).WillOnce(::testing::Return(true));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mock_plan_builder));

    MockGraph mock_graph;

    EXPECT_TRUE(engine.isApplicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    EXPECT_CALL(*mock_plan_builder1, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder2, isApplicable(::testing::_)).Times(0);

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mock_plan_builder1));
    engine.addPlanBuilder(std::move(mock_plan_builder2));

    MockGraph mock_graph;

    EXPECT_TRUE(engine.isApplicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    MiopenEngine engine(0);

    MockGraph mock_graph;

    EXPECT_FALSE(engine.isApplicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, isApplicable(::testing::_)).WillOnce(::testing::Return(false));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mock_plan_builder));

    MockGraph mock_graph;

    EXPECT_FALSE(engine.isApplicable(mock_graph));
}

TEST(Miopen_engineTest, GetDetailsReturnsSerializedEngineDetails)
{
    MiopenEngine engine(1);
    HipdnnEnginePluginHandle dummy_handle;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummy_handle, result);

    hipdnn_plugin::EngineDetailsWrapper engine_details(result.ptr, result.size);
    EXPECT_EQ(engine_details.engineId(), 1);
}

TEST(Miopen_engineTest, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mock_plan_builder1, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mock_plan_builder2, isApplicable(::testing::_)).Times(0);
    EXPECT_CALL(*mock_plan_builder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mock_plan_builder1));
    engine.addPlanBuilder(std::move(mock_plan_builder2));

    MockGraph mock_graph;
    HipdnnEnginePluginHandle dummy_handle;
    Mock_hipdnn_engine_plugin_execution_context ctx;

    engine.initializeExecutionContext(dummy_handle, mock_graph, ctx);
}

TEST(Miopen_engineTest, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mock_plan_builder1, isApplicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_plan_builder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_plan_builder2, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mock_plan_builder1));
    engine.addPlanBuilder(std::move(mock_plan_builder2));

    MockGraph mock_graph;
    HipdnnEnginePluginHandle dummy_handle;
    Mock_hipdnn_engine_plugin_execution_context ctx;

    engine.initializeExecutionContext(dummy_handle, mock_graph, ctx);
}

TEST(Miopen_engineTest, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    EXPECT_CALL(*mock_plan_builder1, isApplicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_plan_builder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_plan_builder2, isApplicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_plan_builder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mock_plan_builder1));
    engine.addPlanBuilder(std::move(mock_plan_builder2));

    MockGraph mock_graph;
    HipdnnEnginePluginHandle dummy_handle;
    Mock_hipdnn_engine_plugin_execution_context ctx;

    engine.initializeExecutionContext(dummy_handle, mock_graph, ctx);
}
