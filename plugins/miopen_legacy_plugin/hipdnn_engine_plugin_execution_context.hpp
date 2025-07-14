// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_config_wrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

struct hipdnnEnginePluginExecutionContext
{
public:
    hipdnnEnginePluginExecutionContext(const hipdnnPluginConstData_t* engine_config_ptr,
                                       const hipdnnPluginConstData_t* op_graph_ptr)
    {
        if(engine_config_ptr == nullptr || op_graph_ptr == nullptr)
        {
            _engine_config.ptr = nullptr;
            _engine_config.size = 0;
            _op_graph.ptr = nullptr;
            _op_graph.size = 0;
            return;
        }

        auto* temp_buffer = new uint8_t[engine_config_ptr->size];
        std::memcpy(temp_buffer, engine_config_ptr->ptr, engine_config_ptr->size);
        _engine_config = hipdnnPluginConstData_t{temp_buffer, engine_config_ptr->size};

        temp_buffer = new uint8_t[op_graph_ptr->size];
        std::memcpy(temp_buffer, op_graph_ptr->ptr, op_graph_ptr->size);
        _op_graph = hipdnnPluginConstData_t{temp_buffer, op_graph_ptr->size};

        _graph_interface
            = std::make_unique<hipdnn_plugin::Graph_wrapper>(_op_graph.ptr, _op_graph.size);

        _engine_config_interface = std::make_unique<hipdnn_plugin::Engine_config_wrapper>(
            _engine_config.ptr, _engine_config.size);
    }

    virtual ~hipdnnEnginePluginExecutionContext()
    {
        _graph_interface.reset();
        _engine_config_interface.reset();

        if(_engine_config.ptr != nullptr)
        {
            delete[] static_cast<const uint8_t*>(_engine_config.ptr);
        }
        if(_op_graph.ptr != nullptr)
        {
            delete[] static_cast<const uint8_t*>(_op_graph.ptr);
        }
    }

    virtual hipdnn_plugin::Graph_interface& graph() const
    {
        return *_graph_interface;
    }

    virtual hipdnn_plugin::Engine_config_interface& engine_config() const
    {
        return *_engine_config_interface;
    }

private:
    hipdnnPluginConstData_t _engine_config;
    hipdnnPluginConstData_t _op_graph;

    std::unique_ptr<hipdnn_plugin::Graph_wrapper> _graph_interface;
    std::unique_ptr<hipdnn_plugin::Engine_config_wrapper> _engine_config_interface;
};