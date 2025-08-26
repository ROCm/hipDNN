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

TEST(MiopenEngineTest, ConstructorAndId)
{
    MiopenEngine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(MiopenEngineTest, WorkspaceSizeReturnsZeroIfNoPlanBuilders)
{
    MiopenEngine engine(1);

    Mock_graph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(MiopenEngineTest, WorkspaceSizeReturnsPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    Mock_graph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 1337u);
}

TEST(MiopenEngineTest, WorkspaceSizeReturnsMaxPlanBuilderWorkspace)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, getWorkspaceSize(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(45000u));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Mock_graph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 45000u);
}

TEST(MiopenEngineTest, WorkspaceSizeReturnsZeroIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_)).WillOnce(::testing::Return(false));

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    Mock_graph mockGraph;

    HipdnnEnginePluginHandle dummyHandle;
    EXPECT_EQ(engine.getWorkspaceSize(dummyHandle, mockGraph), 0u);
}

TEST(MiopenEngineTest, IsApplicableReturnsTrueIfAnyPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_)).WillOnce(::testing::Return(true));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    Mock_graph mockGraph;

    EXPECT_TRUE(engine.isApplicable(mockGraph));
}

TEST(MiopenEngineTest, IsApplicableReturnsAfterTheFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_)).Times(0);

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Mock_graph mockGraph;

    EXPECT_TRUE(engine.isApplicable(mockGraph));
}

TEST(MiopenEngineTest, IsApplicableReturnsFalseIfNoPlanBuilders)
{
    MiopenEngine engine(0);

    Mock_graph mockGraph;

    EXPECT_FALSE(engine.isApplicable(mockGraph));
}

TEST(MiopenEngineTest, IsApplicableReturnsFalseIfNoPlanBuilderApplicable)
{
    auto mockPlanBuilder = std::make_unique<MockPlanBuilder>();
    EXPECT_CALL(*mockPlanBuilder, isApplicable(::testing::_)).WillOnce(::testing::Return(false));

    MiopenEngine engine(0);
    engine.addPlanBuilder(std::move(mockPlanBuilder));

    Mock_graph mockGraph;

    EXPECT_FALSE(engine.isApplicable(mockGraph));
}

TEST(MiopenEngineTest, GetDetailsReturnsSerializedEngineDetails)
{
    MiopenEngine engine(1);
    HipdnnEnginePluginHandle dummyHandle;

    hipdnnPluginConstData_t result;
    engine.getDetails(dummyHandle, result);

    hipdnn_plugin::Engine_details_wrapper engineDetails(result.ptr, result.size);
    EXPECT_EQ(engineDetails.engine_id(), 1);
}

TEST(MiopenEngineTest, InitializeExecutionContextInvokesFirstApplicablePlanBuilder)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // Only the first plan builder is applicable
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Mock_graph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}

TEST(MiopenEngineTest, InitializeExecutionContextSkipsNonApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    // First plan builder not applicable, second is
    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(1);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Mock_graph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}

TEST(MiopenEngineTest, InitializeExecutionContextDoesNotCallBuildPlanIfNoApplicableBuilders)
{
    auto mockPlanBuilder1 = std::make_unique<MockPlanBuilder>();
    auto mockPlanBuilder2 = std::make_unique<MockPlanBuilder>();

    EXPECT_CALL(*mockPlanBuilder1, isApplicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder1, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mockPlanBuilder2, isApplicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockPlanBuilder2, buildPlan(::testing::_, ::testing::_, ::testing::_)).Times(0);

    MiopenEngine engine(1);
    engine.addPlanBuilder(std::move(mockPlanBuilder1));
    engine.addPlanBuilder(std::move(mockPlanBuilder2));

    Mock_graph mockGraph;
    HipdnnEnginePluginHandle dummyHandle;
    MockHipdnnEnginePluginExecutionContext ctx;

    engine.initializeExecutionContext(dummyHandle, mockGraph, ctx);
}
