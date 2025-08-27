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

class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    virtual bool isApplicable(const hipdnn_plugin::IGraph& opGraph) const = 0;

    virtual size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin::IGraph& opGraph) const
        = 0;

    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_plugin::IGraph& opGraph,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;
};
}
