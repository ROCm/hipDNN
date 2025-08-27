// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plan_builder_interface.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class MiopenBatchnormPlanBuilder : public PlanBuilderInterface
{
public:
    MiopenBatchnormPlanBuilder() = default;
    ~MiopenBatchnormPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenBatchnormPlanBuilder(const MiopenBatchnormPlanBuilder&) = delete;
    MiopenBatchnormPlanBuilder& operator=(const MiopenBatchnormPlanBuilder&) = delete;

    bool isApplicable(const hipdnn_plugin::IGraph& opGraph) const override;
    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_plugin::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;
};

}
