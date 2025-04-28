// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plugin/engine_plugin.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <numeric>
#include <vector>

using namespace hipdnn_backend;

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST(EnginePluginManagerTest, LoadPluginsAndRunEngines)
{
    // Skip the test if no devices are available.
    int device_count;
    // hipGetDeviceCount() can also return hipErrorNoDevice if no devices are available.
    // This is not a failure, so we should not use ASSERT_EQ here.
    auto result = hipGetDeviceCount(&device_count);
    if(result == hipErrorNoDevice)
    {
        GTEST_SKIP() << "No devices available. Skipping test.";
    }
    ASSERT_EQ(result, hipSuccess);

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

    // The number of elements in the input vector.
    const unsigned data_size = 512;

    // The number of bytes to allocate for the input- and output device vectors.
    const std::size_t size_bytes = data_size * sizeof(uint32_t);

    // Allocate host input buffer and fill it with an increasing sequence (i.e. 0, 1, 2, ...).
    std::vector<uint32_t> in_host_data(data_size);
    std::iota(in_host_data.begin(), in_host_data.end(), 0);

    // Host output buffer.
    std::vector<uint32_t> out_host_data(data_size);

    // Allocate input and output device buffers.
    uint32_t* in_dev_data{};
    uint32_t* out_dev_data{};
    ASSERT_EQ(hipMalloc(&in_dev_data, size_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&out_dev_data, size_bytes), hipSuccess);

    // Copy the input data from the host to the device.
    ASSERT_EQ(hipMemcpy(in_dev_data, in_host_data.data(), size_bytes, hipMemcpyHostToDevice),
              hipSuccess);

    // Call run_engine() on each engine of each plugin
    for(const auto& plugin : plugins)
    {
        const auto num_engines = plugin.num_engines();
        ASSERT_GT(num_engines, 0); // Ensure at least one engine is available

        for(unsigned engine_index = 0; engine_index < num_engines; ++engine_index)
        {
            // Fill output device buffer with zeros
            ASSERT_EQ(hipMemset(out_dev_data, 0, size_bytes), hipSuccess);

            // Run the engine
            plugin.run_engine(engine_index, in_dev_data, out_dev_data, data_size);

            // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
            ASSERT_EQ(
                hipMemcpy(out_host_data.data(), out_dev_data, size_bytes, hipMemcpyDeviceToHost),
                hipSuccess);

            // Check the results
            ASSERT_EQ(std::equal(in_host_data.begin(),
                                 in_host_data.end(),
                                 out_host_data.begin(),
                                 out_host_data.end()),
                      true);
        }
    }

    // Free the device memory
    ASSERT_EQ(hipFree(in_dev_data), hipSuccess);
    ASSERT_EQ(hipFree(out_dev_data), hipSuccess);
}
// NOLINTEND(readability-function-cognitive-complexity)
