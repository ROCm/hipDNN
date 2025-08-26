// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>

#include "mocks/mock_plan.hpp"

#include "hipdnn_engine_plugin_execution_context.hpp"

using namespace miopen_legacy_plugin;

TEST(HipdnnEnginePluginExecutionContextTest, SetAndGetPlan)
{
    HipdnnEnginePluginExecutionContext ctx;

    auto mock_plan = std::make_unique<miopen_legacy_plugin::Mock_plan>();
    auto* plan_ptr = mock_plan.get();
    ctx.setPlan(std::move(mock_plan));

    miopen_legacy_plugin::PlanInterface& plan_ref = ctx.plan();

    EXPECT_EQ(&plan_ref, plan_ptr);
}

TEST(HipdnnEnginePluginExecutionContextTest, HasValidPlan)
{
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_FALSE(ctx.hasValidPlan());

    auto mock_plan = std::make_unique<miopen_legacy_plugin::Mock_plan>();
    ctx.setPlan(std::move(mock_plan));

    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST(HipdnnEnginePluginExecutionContextTest, GetPlanThrowsIfNotSet)
{
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(ctx.plan(), hipdnn_plugin::Hipdnn_plugin_exception);
}
