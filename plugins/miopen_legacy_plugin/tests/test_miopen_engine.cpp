// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/miopen_engine.hpp"
#include "mocks/mock_solver.hpp"

#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <memory>
#include <set>

using namespace miopen_legacy_plugin;

TEST(Miopen_engineTest, ConstructorAndId)
{
    Miopen_engine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(Miopen_engineTest, WorkspaceSize)
{
    Miopen_engine engine(1);
    EXPECT_EQ(engine.get_workspace_size(), 1337);
}

TEST(Miopen_engineTest, IsApplicableReturnsTrueIfAnySolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(true));

    Miopen_engine engine(0);
    engine.add_solver(std::move(mock_solver));

    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    EXPECT_TRUE(engine.is_applicable(&op_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoSolvers)
{
    Miopen_engine engine(0);

    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    EXPECT_FALSE(engine.is_applicable(&op_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoSolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(false));

    Miopen_engine engine(0);
    engine.add_solver(std::move(mock_solver));

    auto builder = flatbuffer_test_utils::create_valid_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    EXPECT_FALSE(engine.is_applicable(&op_graph));
}
