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
#include "mocks/mock_solver.hpp"

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;

TEST(Miopen_engineTest, ConstructorAndId)
{
    Miopen_engine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoSolvers)
{
    Miopen_engine engine(1);

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 0u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsSolverWorkspace)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_solver, get_workspace_size(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 1337u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoSolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(false));

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    Mock_graph mock_graph;

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, mock_graph), 0u);
}

TEST(Miopen_engineTest, IsApplicableReturnsTrueIfAnySolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(true));

    Miopen_engine engine(0);
    engine.add_solver(std::move(mock_solver));

    Mock_graph mock_graph;

    EXPECT_TRUE(engine.is_applicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoSolvers)
{
    Miopen_engine engine(0);

    Mock_graph mock_graph;

    EXPECT_FALSE(engine.is_applicable(mock_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoSolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(false));

    Miopen_engine engine(0);
    engine.add_solver(std::move(mock_solver));

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

TEST(Miopen_engineTest, ExecuteGraphCallsSolverIfApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    // Solver is applicable, so execute_graph should be called
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_solver,
                execute_graph(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    hipdnnEnginePluginHandle dummy_handle = {};
    Mock_hipdnn_engine_plugin_execution_context exec_ctx;

    hipdnnPluginDeviceBuffer_t* device_buffers = nullptr;
    uint32_t num_device_buffers = 0;
    void* workspace = nullptr;

    engine.execute_graph(dummy_handle, exec_ctx, device_buffers, num_device_buffers, workspace);
}

TEST(Miopen_engineTest, ExecuteGraphDoesNotCallSolverIfNotApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    // Solver is not applicable, so execute_graph should not be called
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mock_solver,
                execute_graph(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    hipdnnEnginePluginHandle dummy_handle = {};
    Mock_hipdnn_engine_plugin_execution_context exec_ctx;
    hipdnnPluginDeviceBuffer_t* device_buffers = nullptr;
    uint32_t num_device_buffers = 0;
    void* workspace = nullptr;

    engine.execute_graph(dummy_handle, exec_ctx, device_buffers, num_device_buffers, workspace);
}
