// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "hipdnn_backend.h"

namespace hipdnn_sdk
{
namespace data_objects
{
struct EngineDetails;
}
}

namespace hipdnn_backend
{

class Graph_descriptor;

namespace plugin
{

class Engine_plugin;
class Engine_plugin_manager;
class Engine_details_wrapper;
class Engine_execution_context_wrapper;

class Engine_plugin_resource_manager
{
public:
    // MT-safe static functions
    // Load plugins from a specific path, for testing purposes
    static void set_plugin_paths(const std::vector<std::filesystem::path>& plugin_paths);
    static std::shared_ptr<Engine_plugin_resource_manager> create();

    Engine_plugin_resource_manager(std::shared_ptr<Engine_plugin_manager>& pm);
    ~Engine_plugin_resource_manager();

    // Prevent copying
    Engine_plugin_resource_manager(const Engine_plugin_resource_manager&) = delete;
    Engine_plugin_resource_manager& operator=(const Engine_plugin_resource_manager&) = delete;

    // Allow moving
    Engine_plugin_resource_manager(Engine_plugin_resource_manager&& other) noexcept;
    Engine_plugin_resource_manager& operator=(Engine_plugin_resource_manager&& other) noexcept;

    // MT-unsafe instance methods
    void set_stream(hipStream_t stream) const;

    // TODO: Move to the descriptors
    // This will be moved to the descriptors in the integration stage
#if 0
    void finalize_engine(hipdnnBackendDescriptor_t desc) const;
    void finalize_engine_config(hipdnnBackendDescriptor_t desc) const;
    void finalize_engine_heuristic(hipdnnBackendDescriptor_t desc) const;
    void finalize_execution_plan(hipdnnBackendDescriptor_t desc) const;
#endif

    void execute_op_graph(hipdnnBackendDescriptor_t execution_plan,
                          hipdnnBackendDescriptor_t variant_pack) const;

private:
    // MT-unsafe instance methods
    std::vector<int64_t> get_applicable_engine_ids(Graph_descriptor* graph_desc) const;

    void get_engine_details(int64_t engine_id,
                            Graph_descriptor* graph_desc,
                            hipdnnPluginConstData_t* engine_details) const;
    void destroy_engine_details(int64_t engine_id, hipdnnPluginConstData_t* engine_details) const;
    static std::unique_ptr<Engine_details_wrapper>
        get_engine_details(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                           int64_t engine_id,
                           Graph_descriptor* graph_desc);

    size_t get_workspace_size(int64_t engine_id,
                              const hipdnnPluginConstData_t* engine_config,
                              Graph_descriptor* graph_desc) const;

    hipdnnEnginePluginExecutionContext_t
        create_execution_context(int64_t engine_id,
                                 const hipdnnPluginConstData_t* engine_config,
                                 Graph_descriptor* graph_desc) const;
    void destroy_execution_context(int64_t engine_id,
                                   hipdnnEnginePluginExecutionContext_t execution_context) const;
    static std::unique_ptr<Engine_execution_context_wrapper>
        create_execution_context(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                                 int64_t engine_id,
                                 const hipdnnPluginConstData_t* engine_config,
                                 Graph_descriptor* graph_desc);

    void execute_op_graph(int64_t engine_id,
                          hipdnnEnginePluginExecutionContext_t execution_context,
                          void* workspace,
                          const hipdnnPluginDeviceBuffer_t* device_buffers,
                          uint32_t num_device_buffers) const;

    std::shared_ptr<Engine_plugin_manager> _pm;
    std::unordered_map<hipdnnEnginePluginHandle_t, const Engine_plugin*> _handle_to_plugin;
    mutable std::unordered_map<int64_t, hipdnnEnginePluginHandle_t> _engine_id_to_handle;

    friend class Engine_details_wrapper;
    friend class Engine_execution_context_wrapper;
};

// A class to manage engine details lifecycle
class Engine_details_wrapper
{
public:
    Engine_details_wrapper(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                           int64_t engine_id,
                           Graph_descriptor* graph_desc);
    ~Engine_details_wrapper();

    // Prevent copying
    Engine_details_wrapper(const Engine_details_wrapper&) = delete;
    Engine_details_wrapper& operator=(const Engine_details_wrapper&) = delete;

    // Allow moving
    Engine_details_wrapper(Engine_details_wrapper&& other) noexcept;
    Engine_details_wrapper& operator=(Engine_details_wrapper&& other) noexcept;

    const hipdnn_sdk::data_objects::EngineDetails* get() const;

private:
    std::shared_ptr<Engine_plugin_resource_manager> _rm;
    hipdnnPluginConstData_t _engine_details_data;
};

// A class to manage engine execution context lifecycle
class Engine_execution_context_wrapper
{
public:
    Engine_execution_context_wrapper(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                                     int64_t engine_id,
                                     const hipdnnPluginConstData_t* engine_config,
                                     Graph_descriptor* graph_desc);
    ~Engine_execution_context_wrapper();

    // Prevent copying
    Engine_execution_context_wrapper(const Engine_execution_context_wrapper&) = delete;
    Engine_execution_context_wrapper& operator=(const Engine_execution_context_wrapper&) = delete;

    // Allow moving
    Engine_execution_context_wrapper(Engine_execution_context_wrapper&& other) noexcept;
    Engine_execution_context_wrapper& operator=(Engine_execution_context_wrapper&& other) noexcept;

    hipdnnEnginePluginExecutionContext_t get() const;

private:
    std::shared_ptr<Engine_plugin_resource_manager> _rm;
    int64_t _engine_id;
    hipdnnEnginePluginExecutionContext_t _execution_context;
};

} // namespace plugin
} // hipdnn_backend
