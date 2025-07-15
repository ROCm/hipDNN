// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/scoped_resource.hpp>

#include "plugin/engine_plugin.hpp"

using namespace hipdnn_backend;

template <typename T, typename Destructor>
using Scoped_resource = hipdnn::sdk::utilities::Scoped_resource<T, Destructor>;

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST(GPU_EnginePluginManagerTest, LoadPluginsAndExecuteOpGraph)
{
    SKIP_IF_NO_DEVICES();

    // Create an EngienPluginManager instance
    plugin::Engine_plugin_manager plugin_manager;

    // Create a list of paths to plugins
    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_engine_plugin1"};

    // Load the plugins
    plugin_manager.load_plugins(plugin_paths);

    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 1); // Ensure one plugin is loaded

    // Check that the plugins have the correct names
    ASSERT_EQ(plugins[0].name(), "EnginePlugin1");

    // Check that the plugins have the correct versions
    ASSERT_EQ(plugins[0].version(), "1.0");

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    Scoped_resource stream_res(stream, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

    // TODO set a real op graph
    const std::array<uint8_t, 8> op_graph_data = {0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
    const hipdnnPluginConstData_t op_graph
        = {.ptr = op_graph_data.data(), .size = op_graph_data.size()};

    // The number of elements in the input vector.
    const unsigned data_size = 512;

    // The number of bytes to allocate for the input- and output device vectors.
    const std::size_t size_bytes = data_size * sizeof(uint32_t);

    // Allocate host input buffer and fill it with an increasing sequence (i.e. 0, 1, 2, ...).
    std::vector<uint32_t> in_host_data(data_size);
    std::iota(in_host_data.begin(), in_host_data.end(), 0);

    // Host output buffer.
    std::vector<uint32_t> out_host_data(data_size);

    // Allocate input device buffer.
    uint32_t* in_dev_data{};
    ASSERT_EQ(hipMalloc(&in_dev_data, size_bytes), hipSuccess);
    Scoped_resource in_dev_data_res(in_dev_data, [](uint32_t* ptr) { std::ignore = hipFree(ptr); });

    // Allocate output device buffer.
    uint32_t* out_dev_data{};
    ASSERT_EQ(hipMalloc(&out_dev_data, size_bytes), hipSuccess);
    Scoped_resource out_dev_data_res(out_dev_data,
                                     [](uint32_t* ptr) { std::ignore = hipFree(ptr); });

    // Copy the input data from the host to the device.
    ASSERT_EQ(hipMemcpy(in_dev_data, in_host_data.data(), size_bytes, hipMemcpyHostToDevice),
              hipSuccess);

    for(const auto& plugin : plugins)
    {
        auto handle = plugin.create_handle();
        Scoped_resource handle_res(handle, [&plugin](auto h) { plugin.destroy_handle(h); });

        plugin.set_stream(handle, stream);

        const auto engine_ids = plugin.get_applicable_engine_ids(handle, &op_graph);
        ASSERT_GT(engine_ids.size(), 0); // Ensure at least one engine is applicable

        for(const auto engine_id : engine_ids)
        {
            // Get engine details
            hipdnnPluginConstData_t engine_details;
            plugin.get_engine_details(handle, engine_id, &op_graph, &engine_details);
            Scoped_resource engine_details_res(&engine_details,
                                               [handle, &plugin](hipdnnPluginConstData_t* ed) {
                                                   plugin.destroy_engine_details(handle, ed);
                                               });

            // Prepare the engine configuration
            // TODO set a real engine config based on the engine details
            const hipdnnPluginConstData_t engine_config = {.ptr = nullptr, .size = 0};

            // Create workspace for the operation
            auto workspace_size = plugin.get_workspace_size(handle, &engine_config, &op_graph);
            void* workspace = nullptr;
            if(workspace_size > 0)
            {
                ASSERT_EQ(hipMalloc(&workspace, workspace_size), hipSuccess);
            }
            Scoped_resource workspace_res(workspace, [](void* ptr) { std::ignore = hipFree(ptr); });

            // Create execution context for the operation
            auto execution_context
                = plugin.create_execution_context(handle, &engine_config, &op_graph);
            Scoped_resource execution_context_res(execution_context, [&plugin, handle](auto ec) {
                plugin.destroy_execution_context(handle, ec);
            });

            // Fill output device buffer with zeros
            ASSERT_EQ(hipMemset(out_dev_data, 0, size_bytes), hipSuccess);

            // Prepare device buffers structure
            const uint32_t num_device_buffers = 2;
            const std::array<hipdnnPluginDeviceBuffer_t, num_device_buffers> device_buffers
                = {{{.uid = 0, .ptr = in_dev_data}, {.uid = 1, .ptr = out_dev_data}}};

            // Execute the operation graph
            plugin.execute_op_graph(
                handle, execution_context, workspace, device_buffers.data(), num_device_buffers);

            // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
            ASSERT_EQ(
                hipMemcpy(out_host_data.data(), out_dev_data, size_bytes, hipMemcpyDeviceToHost),
                hipSuccess);

            // Check the results
            ASSERT_EQ(std::ranges::equal(in_host_data.begin(),
                                         in_host_data.end(),
                                         out_host_data.begin(),
                                         out_host_data.end()),
                      true);
        }
    }
}
// NOLINTEND(readability-function-cognitive-complexity)
