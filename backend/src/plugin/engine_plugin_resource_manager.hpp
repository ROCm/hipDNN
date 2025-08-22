// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "hipdnn_backend.h"

namespace hipdnn_sdk
{
namespace data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
struct EngineDetails;
}
}

namespace hipdnn_backend
{

class Graph_descriptor;

namespace plugin
{

class Engine_details_wrapper;
class Engine_execution_context_wrapper;
class Engine_plugin;
class Engine_plugin_manager;

class Engine_plugin_resource_manager
{
protected:
    // Protected constructor for mock testing
    Engine_plugin_resource_manager();

public:
    // MT-safe static functions
    // Load plugins from a specific path, for testing purposes
    static void set_plugin_paths(const std::vector<std::filesystem::path>& plugin_paths,
                                 hipdnnPluginLoadingMode_ext_t loading_mode);
    static std::set<std::filesystem::path> get_plugin_paths();

    static std::shared_ptr<Engine_plugin_resource_manager> create();

    Engine_plugin_resource_manager(std::shared_ptr<Engine_plugin_manager> pm);
    virtual ~Engine_plugin_resource_manager();

    // Prevent copying
    Engine_plugin_resource_manager(const Engine_plugin_resource_manager&) = delete;
    Engine_plugin_resource_manager& operator=(const Engine_plugin_resource_manager&) = delete;

    // Allow moving
    Engine_plugin_resource_manager(Engine_plugin_resource_manager&& other) noexcept;
    Engine_plugin_resource_manager& operator=(Engine_plugin_resource_manager&& other) noexcept;

    // MT-unsafe instance methods
    // virtual for gMock testing
    virtual void set_stream(hipStream_t stream) const;
    virtual std::vector<int64_t>
        get_applicable_engine_ids(const Graph_descriptor* graph_desc) const;
    virtual size_t get_workspace_size(int64_t engine_id,
                                      const hipdnnPluginConstData_t* engine_config,
                                      const Graph_descriptor* graph_desc) const;

    virtual void execute_op_graph(hipdnnBackendDescriptor_t execution_plan,
                                  hipdnnBackendDescriptor_t variant_pack) const;

    static std::shared_ptr<const Engine_details_wrapper>
        get_engine_details(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                           int64_t engine_id,
                           const Graph_descriptor* graph_desc);
    static std::shared_ptr<const Engine_execution_context_wrapper>
        create_execution_context(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                                 int64_t engine_id,
                                 const hipdnnPluginConstData_t* engine_config,
                                 const Graph_descriptor* graph_desc);

    virtual void get_loaded_plugin_files(size_t* num_plugins,
                                         char** plugin_paths,
                                         size_t* max_string_len) const;

private:
    // MT-unsafe instance methods
    // virtual for gMock testing
    virtual void get_engine_details(int64_t engine_id,
                                    const Graph_descriptor* graph_desc,
                                    hipdnnPluginConstData_t* engine_details) const;
    virtual void destroy_engine_details(int64_t engine_id,
                                        hipdnnPluginConstData_t* engine_details) const;

    [[nodiscard]] virtual hipdnnEnginePluginExecutionContext_t
        create_execution_context(int64_t engine_id,
                                 const hipdnnPluginConstData_t* engine_config,
                                 const Graph_descriptor* graph_desc) const;
    virtual void
        destroy_execution_context(int64_t engine_id,
                                  hipdnnEnginePluginExecutionContext_t execution_context) const;

    void execute_op_graph(int64_t engine_id,
                          hipdnnEnginePluginExecutionContext_t execution_context,
                          void* workspace,
                          const hipdnnPluginDeviceBuffer_t* device_buffers,
                          uint32_t num_device_buffers) const;

    std::shared_ptr<Engine_plugin_manager> _pm;
    std::unordered_map<hipdnnEnginePluginHandle_t, const Engine_plugin*> _handle_to_plugin;
    std::unordered_map<int64_t, hipdnnEnginePluginHandle_t> _engine_id_to_handle;

    friend class Engine_details_wrapper;
    friend class Engine_execution_context_wrapper;
};

// A class to manage engine details lifecycle
class Engine_details_wrapper
{
public:
    Engine_details_wrapper(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                           int64_t engine_id,
                           const Graph_descriptor* graph_desc);
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
                                     const Graph_descriptor* graph_desc);
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
