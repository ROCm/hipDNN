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

    // We need this to allow mocking this class
    Engine_plugin();

public:
    // Functions that don't require a handle (called first)
    virtual std::vector<int64_t> get_all_engine_ids() const;

    // Handle lifecycle functions
    virtual hipdnnEnginePluginHandle_t create_handle() const;
    virtual void set_stream(hipdnnEnginePluginHandle_t handle, hipStream_t stream) const;

    // Engine discovery and configuration functions
    virtual std::vector<int64_t>
        get_applicable_engine_ids(hipdnnEnginePluginHandle_t handle,
                                  const hipdnnPluginConstData_t* op_graph) const;
    virtual void get_engine_details(hipdnnEnginePluginHandle_t handle,
                                    int64_t engine_id,
                                    const hipdnnPluginConstData_t* op_graph,
                                    hipdnnPluginConstData_t* engine_details) const;
    virtual size_t get_workspace_size(hipdnnEnginePluginHandle_t handle,
                                      const hipdnnPluginConstData_t* engine_config,
                                      const hipdnnPluginConstData_t* op_graph) const;

    // Execution functions
    [[nodiscard]] virtual hipdnnEnginePluginExecutionContext_t
        create_execution_context(hipdnnEnginePluginHandle_t handle,
                                 const hipdnnPluginConstData_t* engine_config,
                                 const hipdnnPluginConstData_t* op_graph) const;
    virtual void execute_op_graph(hipdnnEnginePluginHandle_t handle,
                                  hipdnnEnginePluginExecutionContext_t execution_context,
                                  void* workspace,
                                  const hipdnnPluginDeviceBuffer_t* device_buffers,
                                  uint32_t num_device_buffers) const;

    // Cleanup functions (called in reverse order)
    virtual void
        destroy_execution_context(hipdnnEnginePluginHandle_t handle,
                                  hipdnnEnginePluginExecutionContext_t execution_context) const;
    virtual void destroy_engine_details(hipdnnEnginePluginHandle_t handle,
                                        hipdnnPluginConstData_t* engine_details) const;
    virtual void destroy_handle(hipdnnEnginePluginHandle_t handle) const;

    static hipdnnPluginType_t get_plugin_type()
    {
        return HIPDNN_PLUGIN_TYPE_ENGINE;
    }

private:
    void resolve_symbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif

    mutable std::vector<int64_t> _all_engine_ids;

    hipdnnPluginStatus_t (*_func_get_all_engine_ids)(int64_t*, uint32_t, uint32_t*);
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

} // namespace plugin
} // hipdnn_backend
