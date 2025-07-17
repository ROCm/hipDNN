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

struct hipdnnEnginePluginExecutionContext
{
public:
    virtual ~hipdnnEnginePluginExecutionContext() = default;

    bool has_valid_plan() const
    {
        return _plan != nullptr;
    }

    void set_plan(std::unique_ptr<miopen_legacy_plugin::Plan_interface> plan)
    {
        _plan = std::move(plan);
    }

    virtual miopen_legacy_plugin::Plan_interface& plan() const
    {
        if(!has_valid_plan())
        {
            throw hipdnn_plugin::Hipdnn_plugin_exception(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "Cannot get plan in execution context, its not set");
        }
        return *_plan;
    }

private:
    std::unique_ptr<miopen_legacy_plugin::Plan_interface> _plan;
};