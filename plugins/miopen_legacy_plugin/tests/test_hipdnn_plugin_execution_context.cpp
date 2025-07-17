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

TEST(HipdnnEnginePluginExecutionContextTest, HasValidPlan)
{
    hipdnnEnginePluginExecutionContext ctx;

    EXPECT_FALSE(ctx.has_valid_plan());

    auto mock_plan = std::make_unique<miopen_legacy_plugin::Mock_plan>();
    ctx.set_plan(std::move(mock_plan));

    EXPECT_TRUE(ctx.has_valid_plan());
}

TEST(HipdnnEnginePluginExecutionContextTest, GetPlanThrowsIfNotSet)
{
    hipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(ctx.plan(), hipdnn_plugin::Hipdnn_plugin_exception);
}
