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
    MOCK_METHOD(bool, isApplicable, (const hipdnn_plugin::IGraph& op_graph), (const, override));
    MOCK_METHOD(void,
                getDetails,
                (HipdnnEnginePluginHandle & handle, hipdnnPluginConstData_t& details_out),
                (const, override));
    MOCK_METHOD(size_t,
                getWorkspaceSize,
                (const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph),
                (const, override));

    MOCK_METHOD(void,
                initializeExecutionContext,
                (const HipdnnEnginePluginHandle& handle,
                 const hipdnn_plugin::IGraph& opGraph,
                 HipdnnEnginePluginExecutionContext& execution_context),
                (const, override));
};

} // namespace miopen_legacy_plugin
