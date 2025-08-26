// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/engine_plugin.hpp"
#include "plugin/shared_library.hpp"

#include <cstdint>
#include <gmock/gmock.h>
#include <string_view>
#include <vector>

namespace hipdnn_backend
{
namespace plugin
{

class Mock_engine_plugin : public Engine_plugin
{
public:
    // Mock all public methods from Engine_plugin
    MOCK_METHOD(hipdnnEnginePluginHandle_t, create_handle, (), (const));
    MOCK_METHOD(std::vector<int64_t>, get_all_engine_ids, (), (const));
    MOCK_METHOD(void, destroy_handle, (hipdnnEnginePluginHandle_t handle), (const));
    MOCK_METHOD(void, set_stream, (hipdnnEnginePluginHandle_t handle, hipStream_t stream), (const));
    MOCK_METHOD(std::vector<int64_t>,
                get_applicable_engine_ids,
                (hipdnnEnginePluginHandle_t handle, const hipdnnPluginConstData_t* op_graph),
                (const));
    MOCK_METHOD(void,
                get_engine_details,
                (hipdnnEnginePluginHandle_t handle,
                 int64_t engine_id,
                 const hipdnnPluginConstData_t* op_graph,
                 hipdnnPluginConstData_t* engine_details),
                (const));
    MOCK_METHOD(void,
                destroy_engine_details,
                (hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* engine_details),
                (const));
    MOCK_METHOD(size_t,
                get_workspace_size,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnnPluginConstData_t* engine_config,
                 const hipdnnPluginConstData_t* op_graph),
                (const));
    MOCK_METHOD(hipdnnEnginePluginExecutionContext_t,
                create_execution_context,
                (hipdnnEnginePluginHandle_t handle,
                 const hipdnnPluginConstData_t* engine_config,
                 const hipdnnPluginConstData_t* op_graph),
                (const));
    MOCK_METHOD(void,
                destroy_execution_context,
                (hipdnnEnginePluginHandle_t handle,
                 hipdnnEnginePluginExecutionContext_t execution_context),
                (const));
    MOCK_METHOD(void,
                execute_op_graph,
                (hipdnnEnginePluginHandle_t handle,
                 hipdnnEnginePluginExecutionContext_t execution_context,
                 void* workspace,
                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                 uint32_t num_device_buffers),
                (const));

    // Mock inherited methods from Plugin_base
    MOCK_METHOD(std::string_view, name, (), (const));
    MOCK_METHOD(std::string_view, version, (), (const));
    MOCK_METHOD(hipdnnPluginType_t, type, (), (const));
    MOCK_METHOD(hipdnnPluginStatus_t, set_logging_callback, (hipdnnCallback_t callback), (const));
};

} // namespace plugin
} // namespace hipdnn_backend
