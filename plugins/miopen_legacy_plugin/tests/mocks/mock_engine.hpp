/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#pragma once

#include "engines/engine_interface.hpp"
#include <gmock/gmock.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Mock_engine : public Engine_interface
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool, is_applicable, (const hipdnnPluginConstData_t* op_graph), (const, override));
    MOCK_METHOD(void, get_details, (hipdnnPluginConstData_t & details_out), (const, override));
    MOCK_METHOD(size_t,
                get_workspace_size,
                (const hipdnnEnginePluginHandle& handle, const hipdnnPluginConstData_t* op_graph),
                (const, override));

    MOCK_METHOD(void,
                execute_graph,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnnEnginePluginExecutionContext& execution_context,
                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                 uint32_t num_device_buffers,
                 void* workspace),
                (const, override));
};

} // namespace miopen_legacy_plugin