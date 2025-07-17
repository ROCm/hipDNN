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
    Miopen_engine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    Miopen_engine engine(1);

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 0u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder, get_workspace_size(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    Miopen_engine engine(1);
    engine.add_plan_builder(std::move(mock_plan_builder));

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 1337u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    EXPECT_CALL(*mock_plan_builder, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder, get_workspace_size(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));
    EXPECT_CALL(*mock_plan_builder2, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder2, get_workspace_size(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    Miopen_engine engine(1);
    engine.add_plan_builder(std::move(mock_plan_builder));
    engine.add_plan_builder(std::move(mock_plan_builder2));

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 45000u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, is_applicable(::testing::_)).WillOnce(::testing::Return(false));

    Miopen_engine engine(1);
    engine.add_plan_builder(std::move(mock_plan_builder));

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 0u);
}

TEST(Miopen_engineTest, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, is_applicable(::testing::_)).WillOnce(::testing::Return(true));

    Miopen_engine engine(0);
    engine.add_plan_builder(std::move(mock_plan_builder));

    Mock_graph mock_graph;

    EXPECT_TRUE(engine.is_applicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    EXPECT_CALL(*mock_plan_builder1, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder2, is_applicable(::testing::_)).Times(0);

    Miopen_engine engine(0);
    engine.add_plan_builder(std::move(mock_plan_builder1));
    engine.add_plan_builder(std::move(mock_plan_builder2));

    Mock_graph mock_graph;

    EXPECT_TRUE(engine.is_applicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    Miopen_engine engine(0);

    Mock_graph mock_graph;

    EXPECT_FALSE(engine.is_applicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mock_plan_builder = std::make_unique<Mock_plan_builder>();
    EXPECT_CALL(*mock_plan_builder, is_applicable(::testing::_)).WillOnce(::testing::Return(false));

    Miopen_engine engine(0);
    engine.add_plan_builder(std::move(mock_plan_builder));

    Mock_graph mock_graph;

    EXPECT_FALSE(engine.is_applicable(mock_graph));
}

TEST(Miopen_engineTest, GetDetailsReturnsSerializedEngineDetails)
{
    Miopen_engine engine(1);

    hipdnnPluginConstData_t result;
    engine.get_details(result);

    hipdnn_plugin::Engine_details_wrapper engine_details(result.ptr, result.size);
    EXPECT_EQ(engine_details.engine_id(), 1);

    delete[] static_cast<const uint8_t*>(result.ptr);
}

TEST(Miopen_engineTest, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mock_plan_builder1, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder1, build_plan(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mock_plan_builder2, is_applicable(::testing::_)).Times(0);
    EXPECT_CALL(*mock_plan_builder2, build_plan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    Miopen_engine engine(1);
    engine.add_plan_builder(std::move(mock_plan_builder1));
    engine.add_plan_builder(std::move(mock_plan_builder2));

    Mock_graph mock_graph;
    hipdnnEnginePluginHandle dummy_handle;
    Mock_hipdnn_engine_plugin_execution_context ctx;

    engine.initialize_execution_context(dummy_handle, mock_graph, ctx);
}

TEST(Miopen_engineTest, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mock_plan_builder1, is_applicable(::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_plan_builder1, build_plan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_plan_builder2, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_plan_builder2, build_plan(::testing::_, ::testing::_, ::testing::_)).Times(1);

    Miopen_engine engine(1);
    engine.add_plan_builder(std::move(mock_plan_builder1));
    engine.add_plan_builder(std::move(mock_plan_builder2));

    Mock_graph mock_graph;
    hipdnnEnginePluginHandle dummy_handle;
    Mock_hipdnn_engine_plugin_execution_context ctx;

    engine.initialize_execution_context(dummy_handle, mock_graph, ctx);
}

TEST(Miopen_engineTest, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mock_plan_builder1 = std::make_unique<Mock_plan_builder>();
    auto mock_plan_builder2 = std::make_unique<Mock_plan_builder>();

    EXPECT_CALL(*mock_plan_builder1, is_applicable(::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_plan_builder1, build_plan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_plan_builder2, is_applicable(::testing::_))
        .WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_plan_builder2, build_plan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    Miopen_engine engine(1);
    engine.add_plan_builder(std::move(mock_plan_builder1));
    engine.add_plan_builder(std::move(mock_plan_builder2));

    Mock_graph mock_graph;
    hipdnnEnginePluginHandle dummy_handle;
    Mock_hipdnn_engine_plugin_execution_context ctx;

    engine.initialize_execution_context(dummy_handle, mock_graph, ctx);
}
