// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "mocks/mock_plan.hpp"

#include "hipdnn_engine_plugin_execution_context.hpp"

struct MockHipdnnEnginePluginExecutionContext : public HipdnnEnginePluginExecutionContext
{
    MockHipdnnEnginePluginExecutionContext()
        : _mockPlan(std::make_unique<miopen_legacy_plugin::MockPlan>())
    {
    }

    miopen_legacy_plugin::IPlan& plan() const override
    {
        return *_mockPlan;
    }

    std::unique_ptr<miopen_legacy_plugin::MockPlan> _mockPlan;
};
