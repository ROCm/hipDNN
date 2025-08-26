// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>

#include "engines/plans/plan_builder_interface.hpp"

namespace miopen_legacy_plugin
{

class Mock_plan_builder : public PlanBuilderInterface
{
public:
    MOCK_METHOD(bool,
                isApplicable,
                (const hipdnn_plugin::Graph_interface& opGraph),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& opGraph),
                (const, override));

    MOCK_METHOD(void,
                buildPlan,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& opGraph,
                 hipdnnEnginePluginExecutionContext& executionContext),
                (const, override));
};

}
