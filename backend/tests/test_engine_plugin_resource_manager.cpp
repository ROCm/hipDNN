// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/scoped_resource.hpp>

#include "plugin/engine_plugin_resource_manager.hpp"

using namespace hipdnn_backend;

template <typename T, typename Destructor>
using Scoped_resource = hipdnn::sdk::utilities::Scoped_resource<T, Destructor>;

TEST(GPU_EnginePluginResourceManagerTest, LoadPluginsAndExecuteOpGraph)
{
    SKIP_IF_NO_DEVICES();

    // Create a list of paths to plugins
    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_engine_plugin1"};

    // Set the plugin paths
    plugin::Engine_plugin_resource_manager::set_plugin_paths(plugin_paths,
                                                             HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto retrieved_paths = plugin::Engine_plugin_resource_manager::get_plugin_paths();
    std::set<std::filesystem::path> expected_paths(plugin_paths.begin(), plugin_paths.end());

    EXPECT_EQ(retrieved_paths, expected_paths);

    // Create an Engine_plugin_resource_manager instance
    auto resource_manager = plugin::Engine_plugin_resource_manager::create();

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    Scoped_resource stream_res(stream, [](hipStream_t s) { std::ignore = hipStreamDestroy(s); });

    resource_manager->set_stream(stream);

    // TODO: Implement a test for executing an operation graph
}
