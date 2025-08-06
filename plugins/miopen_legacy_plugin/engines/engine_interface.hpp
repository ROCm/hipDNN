// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Engine_interface
{
public:
    virtual ~Engine_interface() = default;

    virtual int64_t id() const = 0;

    virtual bool is_applicable(const hipdnn_plugin::Graph_interface& op_graph) const = 0;
    virtual void get_details(hipdnnEnginePluginHandle& handle,
                             hipdnnPluginConstData_t& details_out) const
        = 0;

    virtual size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                                      const hipdnn_plugin::Graph_interface& op_graph) const
        = 0;

    virtual void
        initialize_execution_context(const hipdnnEnginePluginHandle& handle,
                                     const hipdnn_plugin::Graph_interface& op_graph,
                                     hipdnnEnginePluginExecutionContext& execution_context) const
        = 0;
};

}
