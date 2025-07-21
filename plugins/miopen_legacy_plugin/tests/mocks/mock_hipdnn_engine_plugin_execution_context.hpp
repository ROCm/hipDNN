// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "mocks/mock_plan.hpp"

#include "hipdnn_engine_plugin_execution_context.hpp"

struct Mock_hipdnn_engine_plugin_execution_context : public hipdnnEnginePluginExecutionContext
{
    Mock_hipdnn_engine_plugin_execution_context()
        : _mock_plan(std::make_unique<miopen_legacy_plugin::Mock_plan>())
    {
    }

    miopen_legacy_plugin::Plan_interface& plan() const override
    {
        return *_mock_plan;
    }

    std::unique_ptr<miopen_legacy_plugin::Mock_plan> _mock_plan;
};
