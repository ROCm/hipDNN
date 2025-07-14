// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_sdk/plugin/test_utils/mock_engine_config.hpp>
#include <hipdnn_sdk/plugin/test_utils/mock_graph.hpp>

#include "hipdnn_engine_plugin_execution_context.hpp"

struct Mock_hipdnn_engine_plugin_execution_context : public hipdnnEnginePluginExecutionContext
{
    Mock_hipdnn_engine_plugin_execution_context()
        : hipdnnEnginePluginExecutionContext(nullptr, nullptr)
        , mock_graph(std::make_unique<hipdnn_plugin::Mock_graph>())
        , mock_engine_config(std::make_unique<hipdnn_plugin::Mock_engine_config>())
    {
    }

    ~Mock_hipdnn_engine_plugin_execution_context() override = default;

    hipdnn_plugin::Graph_interface& graph() const override
    {
        return *mock_graph;
    }

    hipdnn_plugin::Engine_config_interface& engine_config() const override
    {
        return *mock_engine_config;
    }

    std::unique_ptr<hipdnn_plugin::Mock_graph> mock_graph;
    std::unique_ptr<hipdnn_plugin::Mock_engine_config> mock_engine_config;
};
