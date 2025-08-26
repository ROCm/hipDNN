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

class MiopenEngine : public EngineInterface
{
public:
    MiopenEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(const hipdnn_plugin::Graph_interface& opGraph) const override;
    void getDetails(HipdnnEnginePluginHandle& handle,
                    hipdnnPluginConstData_t& detailsOut) const override;
    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::Graph_interface& opGraph) const override;

    void initializeExecutionContext(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_plugin::Graph_interface& opGraph,
        HipdnnEnginePluginExecutionContext& executionContext) const override;

    void addPlanBuilder(std::unique_ptr<PlanBuilderInterface> planBuilder);

private:
    int64_t _id;
    std::set<std::unique_ptr<PlanBuilderInterface>> _planBuilders;
};

}
