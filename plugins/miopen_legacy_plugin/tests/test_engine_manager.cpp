// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>

#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/test_utils/mock_engine_config.hpp>
#include <hipdnn_sdk/plugin/test_utils/mock_graph.hpp>

#include "engine_manager.hpp"
#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"
#include "mocks/mock_engine.hpp"
#include "mocks/mock_hipdnn_engine_plugin_execution_context.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;
using ::testing::Return;

TEST(Engine_managerTest, ReturnsApplicableEngineIds)
{
    std::set<std::unique_ptr<EngineInterface>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, isApplicable(::testing::_)).WillRepeatedly(Return(true));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, isApplicable(::testing::_)).WillRepeatedly(Return(false));

    EngineManager manager;
    manager.addEngine(std::move(mock_engine1));
    manager.addEngine(std::move(mock_engine2));

    MockGraph mock_graph;
    auto applicable = manager.getApplicableEngineIds(mock_graph);

    EXPECT_EQ(applicable.size(), 1);
    EXPECT_EQ(applicable[0], 1);
}

TEST(Engine_managerTest, ReturnsMultipleApplicableEngineIds)
{
    std::set<std::unique_ptr<EngineInterface>> engines;

    MockGraph mock_graph;
    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, isApplicable(::testing::_)).WillRepeatedly(Return(true));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, isApplicable(::testing::_)).WillRepeatedly(Return(true));

    EngineManager manager;
    manager.addEngine(std::move(mock_engine1));
    manager.addEngine(std::move(mock_engine2));

    auto applicable = manager.getApplicableEngineIds(mock_graph);

    EXPECT_EQ(applicable.size(), 2);
    EXPECT_TRUE(std::ranges::find(applicable, 1) != applicable.end());
    EXPECT_TRUE(std::ranges::find(applicable, 2) != applicable.end());
}

TEST(Engine_managerTest, ReturnsNoApplicableEngineIds)
{
    std::set<std::unique_ptr<EngineInterface>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, isApplicable(::testing::_)).WillRepeatedly(Return(false));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, isApplicable(::testing::_)).WillRepeatedly(Return(false));

    EngineManager manager;
    manager.addEngine(std::move(mock_engine1));
    manager.addEngine(std::move(mock_engine2));

    MockGraph mock_graph;
    auto applicable = manager.getApplicableEngineIds(mock_graph);

    EXPECT_TRUE(applicable.empty());
}

TEST(Engine_managerTest, ReturnsEngineDetails)
{
    EngineManager manager;

    hipdnnPluginConstData_t engine_details;
    engine_details.ptr = reinterpret_cast<const void*>(0x12345678);
    engine_details.size = 200;
    auto mock_engine = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine, getDetails(::testing::_, ::testing::_))
        .WillOnce(
            [&engine_details](HipdnnEnginePluginHandle& handle, hipdnnPluginConstData_t& out) {
                (void)handle;
                out.ptr = engine_details.ptr;
                out.size = engine_details.size;
            });

    manager.addEngine(std::move(mock_engine));

    MockGraph mock_graph;
    HipdnnEnginePluginHandle dummy_handle = {};
    hipdnnPluginConstData_t details;
    manager.getEngineDetails(dummy_handle, mock_graph, 1, details);

    EXPECT_EQ(details.ptr, engine_details.ptr);
    EXPECT_EQ(details.size, engine_details.size);
}

TEST(Engine_managerTest, ThrowsOnInvalidEngineId)
{
    EngineManager manager;

    MockGraph mock_graph;
    hipdnnPluginConstData_t engine_details;

    HipdnnEnginePluginHandle dummy_handle = {};
    EXPECT_THROW(manager.getEngineDetails(dummy_handle, mock_graph, 999, engine_details),
                 hipdnn_plugin::HipdnnPluginException);
}

TEST(Engine_managerTest, GetWorkspaceSizeReturnsCorrectValue)
{
    EngineManager manager;

    auto mock_engine = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine, id()).WillRepeatedly(Return(42));
    HipdnnEnginePluginHandle dummy_handle = {};
    MockGraph mock_graph;
    EXPECT_CALL(*mock_engine, getWorkspaceSize(::testing::_, ::testing::_)).WillOnce(Return(4096));

    manager.addEngine(std::move(mock_engine));

    size_t workspace_size = manager.getWorkspaceSize(dummy_handle, 42, mock_graph);
    EXPECT_EQ(workspace_size, 4096);
}

TEST(Engine_managerTest, GetWorkspaceSizeThrowsOnInvalidEngineId)
{
    EngineManager manager;
    HipdnnEnginePluginHandle dummy_handle = {};
    MockGraph mock_graph;

    EXPECT_THROW(manager.getWorkspaceSize(dummy_handle, 999, mock_graph),
                 hipdnn_plugin::HipdnnPluginException);
}

TEST(Engine_managerTest, InitializeExecutionContextCallsEngine)
{
    auto mock_engine = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine, id()).WillRepeatedly(Return(7));
    EXPECT_CALL(*mock_engine, initializeExecutionContext(::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    EngineManager manager;
    manager.addEngine(std::move(mock_engine));
    HipdnnEnginePluginHandle dummy_handle = {};
    MockGraph mock_graph;
    MockEngineConfig mockEngineConfig;
    ON_CALL(mockEngineConfig, engineId()).WillByDefault(Return(7));
    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    Mock_hipdnn_engine_plugin_execution_context exec_ctx;

    manager.initializeExecutionContext(dummy_handle, mock_graph, mockEngineConfig, exec_ctx);
}

TEST(Engine_managerTest, InitializeExecutionContextThrowsOnInvalidEngineId)
{
    Mock_hipdnn_engine_plugin_execution_context exec_ctx;
    EngineManager manager;
    HipdnnEnginePluginHandle dummy_handle = {};
    MockGraph mock_graph;
    MockEngineConfig mockEngineConfig;

    EXPECT_CALL(mockEngineConfig, engineId()).Times(testing::AnyNumber()); // Uninteresting call
    EXPECT_THROW(
        manager.initializeExecutionContext(dummy_handle, mock_graph, mockEngineConfig, exec_ctx),
        hipdnn_plugin::HipdnnPluginException);
}
