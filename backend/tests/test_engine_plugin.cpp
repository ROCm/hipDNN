// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plugin/engine_plugin.hpp"
#include <gtest/gtest.h>

using namespace hipdnn_backend;

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST(EnginePluginManagerTest, LoadPlugins)
{
    // Create an EngienPluginManager instance
    plugin::Engine_plugin_manager plugin_manager;

    // Create a list of paths to plugins
    std::vector<std::filesystem::path> plugin_paths = {"./engineplugin1", "./engineplugin2"};

    // Load the plugins
    plugin_manager.load_plugins(plugin_paths);

    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 2); // Ensure two plugins are loaded

    // Check that the plugins have the correct names
    ASSERT_EQ(plugins[0].name(), "EnginePlugin1");
    ASSERT_EQ(plugins[1].name(), "EnginePlugin2");

    // Check that the plugins have the correct versions
    ASSERT_EQ(plugins[0].version(), "1.0");
    ASSERT_EQ(plugins[1].version(), "2.0");

    // Call run_engine() on each engine of each plugin
    for(const auto& plugin : plugins)
    {
        const auto num_engines = plugin.num_engines();
        ASSERT_GT(num_engines, 0); // Ensure at least one engine is available

        for(unsigned engine_index = 0; engine_index < num_engines; ++engine_index)
        {
            ASSERT_NE(plugin.run_engine(engine_index, 1),
                      -1); // Ensure run_engine() returns a valid value
        }
    }
}
// NOLINTEND(readability-function-cognitive-complexity)
