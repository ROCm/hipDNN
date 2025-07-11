// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>

#include "hipdnn_engine_plugin_execution_context.hpp"

struct Dummy_data
{
    std::array<uint8_t, 8> data;
};

TEST(HipdnnEnginePluginExecutionContextTest, NullPointers)
{
    //no crash or throw
    hipdnnEnginePluginExecutionContext ctx(nullptr, nullptr);
}

TEST(HipdnnEnginePluginExecutionContextTest, ValidPointers)
{
    Dummy_data engine_config = {{1, 2, 3, 4, 5, 6, 7, 8}};
    Dummy_data op_graph = {{8, 7, 6, 5, 4, 3, 2, 1}};
    hipdnnPluginConstData_t engine_config_ptr{engine_config.data.data(),
                                              sizeof(engine_config.data)};
    hipdnnPluginConstData_t op_graph_ptr{op_graph.data.data(), sizeof(op_graph.data)};

    hipdnnEnginePluginExecutionContext ctx(&engine_config_ptr, &op_graph_ptr);

    auto& graph = ctx.graph();
    auto& engine = ctx.engine_config();
    (void)graph;
    (void)engine;
}

TEST(HipdnnEnginePluginExecutionContextTest, DestructorCleansUpAndDoesntThrow)
{
    Dummy_data engine_config = {{0}};
    Dummy_data op_graph = {{0}};
    hipdnnPluginConstData_t engine_config_ptr{engine_config.data.data(),
                                              sizeof(engine_config.data)};
    hipdnnPluginConstData_t op_graph_ptr{op_graph.data.data(), sizeof(op_graph.data)};

    auto ctx
        = std::make_unique<hipdnnEnginePluginExecutionContext>(&engine_config_ptr, &op_graph_ptr);

    ctx.reset();
}
