// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plan_builder_interface.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Miopen_batchnorm_plan_builder : public Plan_builder_interface
{
public:
    Miopen_batchnorm_plan_builder() = default;
    ~Miopen_batchnorm_plan_builder() override = default;

    // Disallow copy and assignment
    Miopen_batchnorm_plan_builder(const Miopen_batchnorm_plan_builder&) = delete;
    Miopen_batchnorm_plan_builder& operator=(const Miopen_batchnorm_plan_builder&) = delete;

    bool is_applicable(const hipdnn_plugin::Graph_interface& op_graph) const override;
    size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                              const hipdnn_plugin::Graph_interface& op_graph) const override;

    void build_plan(const hipdnnEnginePluginHandle& handle,
                    const hipdnn_plugin::Graph_interface& op_graph,
                    hipdnnEnginePluginExecutionContext& execution_context) const override;
};

}
