// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_manager.hpp"

#include "mocks/mock_engine.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>

using namespace miopen_legacy_plugin;
using ::testing::Return;

TEST(Engine_managerTest, ReturnsApplicableEngineIds)
{
    std::set<std::unique_ptr<Engine>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, is_applicable(::testing::_)).WillRepeatedly(Return(true));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, is_applicable(::testing::_)).WillRepeatedly(Return(false));

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine1));
    manager.add_engine(std::move(mock_engine2));

    hipdnnPluginConstData_t* dummy_op_graph = nullptr;
    auto applicable = manager.get_applicable_engine_ids(dummy_op_graph);

    EXPECT_EQ(applicable.size(), 1);
    EXPECT_TRUE(applicable.count(1));
    EXPECT_FALSE(applicable.count(2));
}

TEST(Engine_managerTest, ReturnsMultipleApplicableEngineIds)
{
    std::set<std::unique_ptr<Engine>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, is_applicable(::testing::_)).WillRepeatedly(Return(true));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, is_applicable(::testing::_)).WillRepeatedly(Return(true));

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine1));
    manager.add_engine(std::move(mock_engine2));

    hipdnnPluginConstData_t* dummy_op_graph = nullptr;
    auto applicable = manager.get_applicable_engine_ids(dummy_op_graph);

    EXPECT_EQ(applicable.size(), 2);
    EXPECT_TRUE(applicable.count(1));
    EXPECT_TRUE(applicable.count(2));
}

TEST(Engine_managerTest, ReturnsNoApplicableEngineIds)
{
    std::set<std::unique_ptr<Engine>> engines;

    auto mock_engine1 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine1, id()).WillRepeatedly(Return(1));
    EXPECT_CALL(*mock_engine1, is_applicable(::testing::_)).WillRepeatedly(Return(false));

    auto mock_engine2 = std::make_unique<Mock_engine>();
    EXPECT_CALL(*mock_engine2, id()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mock_engine2, is_applicable(::testing::_)).WillRepeatedly(Return(false));

    Engine_manager manager;
    manager.add_engine(std::move(mock_engine1));
    manager.add_engine(std::move(mock_engine2));

    hipdnnPluginConstData_t* dummy_op_graph = nullptr;
    auto applicable = manager.get_applicable_engine_ids(dummy_op_graph);

    EXPECT_TRUE(applicable.empty());
}
