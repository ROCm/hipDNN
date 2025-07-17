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

private:
    std::shared_ptr<Root_engine_plugin_manager> _root_pm;
    std::unordered_map<hipdnnEnginePluginHandle_t, const Engine_plugin*> _plugin_handles;
};

} // namespace plugin
} // hipdnn_backend
