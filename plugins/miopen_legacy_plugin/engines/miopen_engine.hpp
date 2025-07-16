// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>

#include "engine_interface.hpp"
#include "plans/plan_builder_interface.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Miopen_engine : public Engine_interface
{
public:
    Miopen_engine(int64_t id);

    int64_t id() const override;

    bool is_applicable(const hipdnn_plugin::Graph_interface& op_graph) const override;
    void get_details(hipdnnPluginConstData_t& details_out) const override;
    size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                              const hipdnn_plugin::Graph_interface& op_graph) const override;

    void initialize_execution_context(
        const hipdnnEnginePluginHandle& handle,
        const hipdnn_plugin::Graph_interface& op_graph,
        hipdnnEnginePluginExecutionContext& execution_context) const override;

    void add_plan_builder(std::unique_ptr<Plan_builder_interface> plan_builder);

private:
    int64_t _id;
    std::set<std::unique_ptr<Plan_builder_interface>> _plan_builders;
};

}