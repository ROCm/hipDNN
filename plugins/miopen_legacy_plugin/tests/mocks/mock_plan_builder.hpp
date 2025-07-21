// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>

#include "engines/plans/plan_builder_interface.hpp"

namespace miopen_legacy_plugin
{

class Mock_plan_builder : public Plan_builder_interface
{
public:
    MOCK_METHOD(bool,
                is_applicable,
                (const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));
    MOCK_METHOD(size_t,
                get_workspace_size,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));

    MOCK_METHOD(void,
                build_plan,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph,
                 hipdnnEnginePluginExecutionContext& execution_context),
                (const, override));
};

}