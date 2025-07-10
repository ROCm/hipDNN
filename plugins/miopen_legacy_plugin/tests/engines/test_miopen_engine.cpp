// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/miopen_engine.hpp"
#include "mocks/mock_solver.hpp"

#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_flatbuffer_utilities.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <memory>
#include <set>

using namespace miopen_legacy_plugin;

TEST(Miopen_engineTest, ConstructorAndId)
{
    Miopen_engine engine(42);
    EXPECT_EQ(engine.id(), 42);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoSolvers)
{
    Miopen_engine engine(1);

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, &op_graph), 0u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsSolverWorkspace)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mock_solver, get_workspace_size(::testing::_, ::testing::_))
        .WillOnce(::testing::Return(1337u));

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, &op_graph), 1337u);
}

TEST(Miopen_engineTest, WorkspaceSizeReturnsZeroIfNoSolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(false));

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    hipdnnEnginePluginHandle dummy_handle;
    EXPECT_EQ(engine.get_workspace_size(dummy_handle, &op_graph), 0u);
}

TEST(Miopen_engineTest, IsApplicableReturnsTrueIfAnySolverApplicable)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, is_applicable(::testing::_)).WillOnce(::testing::Return(true));

    Miopen_engine engine(0);
    engine.add_solver(std::move(mock_solver));

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    EXPECT_TRUE(engine.is_applicable(&op_graph));
}

TEST(Miopen_engineTest, IsApplicableReturnsFalseIfNoSolvers)
{
    Miopen_engine engine(0);

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
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

    auto builder = flatbuffer_test_utils::create_valid_batchnorm_graph();
    auto serialized_graph = builder.Release();
    hipdnnPluginConstData_t op_graph
        = flatbuffer_test_utils::create_valid_const_data_graph(serialized_graph);

    EXPECT_FALSE(engine.is_applicable(&op_graph));
}

TEST(Miopen_engineTest, GetDetailsReturnsSerializedEngineDetails)
{
    Miopen_engine engine(1);

    hipdnnPluginConstData_t result;
    engine.get_details(result);

    std::unique_ptr<hipdnn_sdk::data_objects::EngineDetailsT> unpacked_engine_details;
    hipdnn_plugin::flatbuffer_utilities::unpack_serialized_engine_details(
        result.ptr, result.size, unpacked_engine_details);
    EXPECT_EQ(unpacked_engine_details->engine_id, 1);

    delete[] static_cast<const uint8_t*>(result.ptr);
}

TEST(Miopen_engineTest, ExecuteGraphCallsSolver)
{
    auto mock_solver = std::make_unique<Mock_solver>();
    EXPECT_CALL(*mock_solver, execute_graph(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1);

    Miopen_engine engine(1);
    engine.add_solver(std::move(mock_solver));

    hipdnnEnginePluginHandle dummy_handle = {};
    hipdnnEnginePluginExecutionContext exec_ctx;
    hipdnnPluginDeviceBuffer_t* device_buffers = nullptr;
    uint32_t num_device_buffers = 0;
    void* workspace = nullptr;

    engine.execute_graph(dummy_handle, exec_ctx, device_buffers, num_device_buffers, workspace);
}
