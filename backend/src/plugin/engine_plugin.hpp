// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin : public Plugin_base
{
protected:
    // The constructor is protected to prevent direct instantiation of the class.
    Engine_plugin(Shared_library&& lib);

public:
    hipdnnEnginePluginHandle_t create_handle() const;
    void destroy_handle(hipdnnEnginePluginHandle_t handle) const;
    void set_stream(hipdnnEnginePluginHandle_t handle, hipStream_t stream) const;
    std::vector<int64_t> get_applicable_engine_ids(hipdnnEnginePluginHandle_t handle,
                                                   const hipdnnPluginConstData_t* op_graph) const;
    void get_engine_details(hipdnnEnginePluginHandle_t handle,
                            int64_t engine_id,
                            const hipdnnPluginConstData_t* op_graph,
                            hipdnnPluginConstData_t* engine_details) const;
    void destroy_engine_details(hipdnnEnginePluginHandle_t handle,
                                hipdnnPluginConstData_t* engine_details) const;
    size_t get_workspace_size(hipdnnEnginePluginHandle_t handle,
                              const hipdnnPluginConstData_t* engine_config,
                              const hipdnnPluginConstData_t* op_graph) const;
    hipdnnEnginePluginExecutionContext_t
        create_execution_context(hipdnnEnginePluginHandle_t handle,
                                 const hipdnnPluginConstData_t* engine_config,
                                 const hipdnnPluginConstData_t* op_graph) const;
    void destroy_execution_context(hipdnnEnginePluginHandle_t handle,
                                   hipdnnEnginePluginExecutionContext_t execution_context) const;
    void execute_op_graph(hipdnnEnginePluginHandle_t handle,
                          hipdnnEnginePluginExecutionContext_t execution_context,
                          void* workspace,
                          const hipdnnPluginDeviceBuffer_t* device_buffers,
                          uint32_t num_device_buffers) const;

private:
    void resolve_symbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif

    hipdnnPluginStatus_t (*_func_create_handle)(hipdnnEnginePluginHandle_t*);
    hipdnnPluginStatus_t (*_func_destroy_handle)(hipdnnEnginePluginHandle_t);
    hipdnnPluginStatus_t (*_func_set_stream)(hipdnnEnginePluginHandle_t, hipStream_t);
    hipdnnPluginStatus_t (*_func_get_applicable_engine_ids)(
        hipdnnEnginePluginHandle_t, const hipdnnPluginConstData_t*, int64_t*, uint32_t, uint32_t*);
    hipdnnPluginStatus_t (*_func_get_engine_details)(hipdnnEnginePluginHandle_t,
                                                     int64_t,
                                                     const hipdnnPluginConstData_t*,
                                                     hipdnnPluginConstData_t*);
    hipdnnPluginStatus_t (*_func_destroy_engine_details)(hipdnnEnginePluginHandle_t,
                                                         hipdnnPluginConstData_t*);
    hipdnnPluginStatus_t (*_func_get_workspace_size)(hipdnnEnginePluginHandle_t,
                                                     const hipdnnPluginConstData_t*,
                                                     const hipdnnPluginConstData_t*,
                                                     size_t*);
    hipdnnPluginStatus_t (*_func_create_execution_context)(hipdnnEnginePluginHandle_t,
                                                           const hipdnnPluginConstData_t*,
                                                           const hipdnnPluginConstData_t*,
                                                           hipdnnEnginePluginExecutionContext_t*);
    hipdnnPluginStatus_t (*_func_destroy_execution_context)(hipdnnEnginePluginHandle_t,
                                                            hipdnnEnginePluginExecutionContext_t);
    hipdnnPluginStatus_t (*_func_execute_op_graph)(hipdnnEnginePluginHandle_t,
                                                   hipdnnEnginePluginExecutionContext_t,
                                                   void*,
                                                   const hipdnnPluginDeviceBuffer_t*,
                                                   uint32_t);

    friend class Plugin_manager_base<Engine_plugin>;
};

using Engine_plugin_manager = Plugin_manager_base<Engine_plugin>;

} // namespace plugin
} // hipdnn_backend
