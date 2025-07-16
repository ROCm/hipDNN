// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include "mocks/mock_plan.hpp"

#include "hipdnn_engine_plugin_execution_context.hpp"

using namespace miopen_legacy_plugin;

TEST(HipdnnEnginePluginExecutionContextTest, SetAndGetPlan)
{
    hipdnnEnginePluginExecutionContext ctx;

    auto mock_plan = std::make_unique<miopen_legacy_plugin::Mock_plan>();
    auto* plan_ptr = mock_plan.get();
    ctx.set_plan(std::move(mock_plan));

    miopen_legacy_plugin::Plan_interface& plan_ref = ctx.plan();

    EXPECT_EQ(&plan_ref, plan_ptr);
}
