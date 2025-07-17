// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin;
class Root_engine_plugin_manager;

class Engine_plugin_manager
{
public:
    // MT-safe static functions
    // Load plugins from a specific path, for testing purposes
    static void set_plugin_paths(const std::vector<std::filesystem::path>& plugin_paths);
    static std::unique_ptr<Engine_plugin_manager> create();
    static void destroy(std::unique_ptr<Engine_plugin_manager>& manager);

    Engine_plugin_manager(std::shared_ptr<Root_engine_plugin_manager>& root_pm);
    ~Engine_plugin_manager();

    // Prevent copying
    Engine_plugin_manager(const Engine_plugin_manager&) = delete;
    Engine_plugin_manager& operator=(const Engine_plugin_manager&) = delete;

    // Allow moving
    Engine_plugin_manager(Engine_plugin_manager&& other) noexcept;
    Engine_plugin_manager& operator=(Engine_plugin_manager&& other) noexcept;

    // MT-unsafe instance methods
    void set_stream(hipStream_t stream) const;
    std::vector<int64_t> get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph) const;
    void get_engine_details(int64_t engine_id,
                            const hipdnnPluginConstData_t* op_graph,
                            hipdnnPluginConstData_t* engine_details) const;
    void destroy_engine_details(int64_t engine_id, hipdnnPluginConstData_t* engine_details) const;
    size_t get_workspace_size(int64_t engine_id,
                              const hipdnnPluginConstData_t* engine_config,
                              const hipdnnPluginConstData_t* op_graph) const;
    hipdnnEnginePluginExecutionContext_t
        create_execution_context(int64_t engine_id,
                                 const hipdnnPluginConstData_t* engine_config,
                                 const hipdnnPluginConstData_t* op_graph) const;
    void destroy_execution_context(int64_t engine_id,
                                   hipdnnEnginePluginExecutionContext_t execution_context) const;
    void execute_op_graph(int64_t engine_id,
                          hipdnnEnginePluginExecutionContext_t execution_context,
                          void* workspace,
                          const hipdnnPluginDeviceBuffer_t* device_buffers,
                          uint32_t num_device_buffers) const;

private:
    std::shared_ptr<Root_engine_plugin_manager> _root_pm;
    std::unordered_map<hipdnnEnginePluginHandle_t, const Engine_plugin*> _handle_to_plugin;
    mutable std::unordered_map<int64_t, hipdnnEnginePluginHandle_t> _engine_id_to_handle;
};

} // namespace plugin
} // hipdnn_backend
