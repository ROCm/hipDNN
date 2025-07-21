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
    std::set<std::unique_ptr<Engine_interface>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, is_applicable(::testing::_)).WillRepeatedly(Return(true));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, is_applicable(::testing::_)).WillRepeatedly(Return(false));

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine1));
    manager.add_engine(std::move(mock_engine2));

    Mock_graph mock_graph;
    auto applicable = manager.get_applicable_engine_ids(mock_graph);

    EXPECT_EQ(applicable.size(), 1);
    EXPECT_EQ(applicable[0], 1);
}

TEST(Engine_managerTest, ReturnsMultipleApplicableEngineIds)
{
    std::set<std::unique_ptr<Engine_interface>> engines;

    Mock_graph mock_graph;
    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, is_applicable(::testing::_)).WillRepeatedly(Return(true));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, is_applicable(::testing::_)).WillRepeatedly(Return(true));

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine1));
    manager.add_engine(std::move(mock_engine2));

    auto applicable = manager.get_applicable_engine_ids(mock_graph);

    EXPECT_EQ(applicable.size(), 2);
    EXPECT_TRUE(std::ranges::find(applicable, 1) != applicable.end());
    EXPECT_TRUE(std::ranges::find(applicable, 2) != applicable.end());
}

TEST(Engine_managerTest, ReturnsNoApplicableEngineIds)
{
    std::set<std::unique_ptr<Engine_interface>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, is_applicable(::testing::_)).WillRepeatedly(Return(false));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, is_applicable(::testing::_)).WillRepeatedly(Return(false));

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine1));
    manager.add_engine(std::move(mock_engine2));

    Mock_graph mock_graph;
    auto applicable = manager.get_applicable_engine_ids(mock_graph);

    EXPECT_TRUE(applicable.empty());
}

TEST(Engine_managerTest, ReturnsEngineDetails)
{
    Engine_manager manager;

    hipdnnPluginConstData_t engine_details;
    engine_details.ptr = reinterpret_cast<const void*>(0x12345678);
    engine_details.size = 200;
    auto mock_engine = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine, get_details(::testing::_))
        .WillOnce([&engine_details](hipdnnPluginConstData_t& out) {
            out.ptr = engine_details.ptr;
            out.size = engine_details.size;
        });

    manager.add_engine(std::move(mock_engine));

    Mock_graph mock_graph;
    hipdnnPluginConstData_t details;
    manager.get_engine_details(mock_graph, 1, details);

    EXPECT_EQ(details.ptr, engine_details.ptr);
    EXPECT_EQ(details.size, engine_details.size);
}

TEST(Engine_managerTest, ThrowsOnInvalidEngineId)
{
    Engine_manager manager;

    Mock_graph mock_graph;
    hipdnnPluginConstData_t engine_details;

    EXPECT_THROW(manager.get_engine_details(mock_graph, 999, engine_details),
                 hipdnn_plugin::Hipdnn_plugin_exception);
}

TEST(Engine_managerTest, GetWorkspaceSizeReturnsCorrectValue)
{
    Engine_manager manager;

    auto mock_engine = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine, id()).WillRepeatedly(Return(42));
    hipdnnEnginePluginHandle dummy_handle = {};
    Mock_graph mock_graph;
    EXPECT_CALL(*mock_engine, get_workspace_size(::testing::_, ::testing::_))
        .WillOnce(Return(4096));

    manager.add_engine(std::move(mock_engine));

    size_t workspace_size = manager.get_workspace_size(dummy_handle, 42, mock_graph);
    EXPECT_EQ(workspace_size, 4096);
}

TEST(Engine_managerTest, GetWorkspaceSizeThrowsOnInvalidEngineId)
{
    Engine_manager manager;
    hipdnnEnginePluginHandle dummy_handle = {};
    Mock_graph mock_graph;

    EXPECT_THROW(manager.get_workspace_size(dummy_handle, 999, mock_graph),
                 hipdnn_plugin::Hipdnn_plugin_exception);
}

TEST(Engine_managerTest, InitializeExecutionContextCallsEngine)
{
    auto mock_engine = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine, id()).WillRepeatedly(Return(7));
    EXPECT_CALL(*mock_engine,
                initialize_execution_context(::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine));
    hipdnnEnginePluginHandle dummy_handle = {};
    Mock_graph mock_graph;
    Mock_engine_config mock_engine_config;
    ON_CALL(mock_engine_config, engine_id()).WillByDefault(Return(7));
    Mock_hipdnn_engine_plugin_execution_context exec_ctx;

    manager.initialize_execution_context(dummy_handle, mock_graph, mock_engine_config, exec_ctx);
}

TEST(Engine_managerTest, InitializeExecutionContextThrowsOnInvalidEngineId)
{
    Mock_hipdnn_engine_plugin_execution_context exec_ctx;
    Engine_manager manager;
    hipdnnEnginePluginHandle dummy_handle = {};
    Mock_graph mock_graph;
    Mock_engine_config mock_engine_config;

    EXPECT_THROW(manager.initialize_execution_context(
                     dummy_handle, mock_graph, mock_engine_config, exec_ctx),
                 hipdnn_plugin::Hipdnn_plugin_exception);
}
