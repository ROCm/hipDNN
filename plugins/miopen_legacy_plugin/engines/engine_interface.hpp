// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class EngineInterface
{
public:
    virtual ~EngineInterface() = default;

    virtual int64_t id() const = 0;

    virtual bool isApplicable(const hipdnn_plugin::Graph_interface& op_graph) const = 0;
    virtual void getDetails(HipdnnEnginePluginHandle& handle,
                            hipdnnPluginConstData_t& details_out) const
        = 0;

    virtual size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin::Graph_interface& op_graph) const
        = 0;

    virtual void
        initializeExecutionContext(const HipdnnEnginePluginHandle& handle,
                                   const hipdnn_plugin::Graph_interface& op_graph,
                                   HipdnnEnginePluginExecutionContext& execution_context) const
        = 0;
};

}
