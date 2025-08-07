// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

namespace miopen_legacy_plugin
{

class Plan_builder_interface
{
public:
    virtual ~Plan_builder_interface() = default;

    virtual bool is_applicable(const hipdnn_plugin::Graph_interface& op_graph) const = 0;

    virtual size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                                      const hipdnn_plugin::Graph_interface& op_graph) const
        = 0;

    virtual void build_plan(const hipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::Graph_interface& op_graph,
                            hipdnnEnginePluginExecutionContext& execution_context) const
        = 0;
};
}
