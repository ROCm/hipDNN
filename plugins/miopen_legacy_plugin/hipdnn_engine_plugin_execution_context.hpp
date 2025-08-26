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

#include "engines/plans/plan_interface.hpp"

struct HipdnnEnginePluginExecutionContext
{
public:
    virtual ~HipdnnEnginePluginExecutionContext() = default;

    bool hasValidPlan() const
    {
        return _plan != nullptr;
    }

    void setPlan(std::unique_ptr<miopen_legacy_plugin::PlanInterface> plan)
    {
        _plan = std::move(plan);
    }

    virtual miopen_legacy_plugin::PlanInterface& plan() const
    {
        if(!hasValidPlan())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "Cannot get plan in execution context, its not set");
        }
        return *_plan;
    }

private:
    std::unique_ptr<miopen_legacy_plugin::PlanInterface> _plan;
};
