/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "engines/engine_interface.hpp"

namespace miopen_legacy_plugin
{

class Mock_engine : public EngineInterface
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool,
                isApplicable,
                (const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));
    MOCK_METHOD(void,
                getDetails,
                (hipdnnEnginePluginHandle & handle, hipdnnPluginConstData_t& details_out),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionContext,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph,
                 hipdnnEnginePluginExecutionContext& execution_context),
                (const, override));
};

} // namespace miopen_legacy_plugin
