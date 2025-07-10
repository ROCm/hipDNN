// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Solver_interface
{
public:
    virtual ~Solver_interface() = default;

    virtual bool is_applicable(const hipdnn_sdk::data_objects::Graph& op_graph) const = 0;

    virtual size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                                      const hipdnn_sdk::data_objects::Graph& graph) const
        = 0;

    virtual void execute_graph(const hipdnnEnginePluginHandle& handle,
                               const hipdnnEnginePluginExecutionContext& execution_context,
                               const hipdnnPluginDeviceBuffer_t* device_buffers,
                               uint32_t num_device_buffers,
                               void* workspace = nullptr) const
        = 0;
};
}