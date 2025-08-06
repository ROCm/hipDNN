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

class Mock_engine : public Engine_interface
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool,
                is_applicable,
                (const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));
    MOCK_METHOD(void,
                get_details,
                (hipdnnEnginePluginHandle & handle, hipdnnPluginConstData_t& details_out),
                (const, override));
    MOCK_METHOD(size_t,
                get_workspace_size,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph),
                (const, override));

    MOCK_METHOD(void,
                initialize_execution_context,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::Graph_interface& op_graph,
                 hipdnnEnginePluginExecutionContext& execution_context),
                (const, override));
};

} // namespace miopen_legacy_plugin
