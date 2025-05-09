// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "plugin/plugin_core.hpp"

using namespace hipdnn_backend;

namespace
{

class Plugin : public plugin::Plugin_base
{
private:
    using Plugin_base::Plugin_base;
    friend class plugin::Plugin_manager_base<Plugin>;
};

} // namespace

TEST(PluginManagerTest, LoadPlugins)
{
    // Create a PluginManager instance
    plugin::Plugin_manager_base<Plugin> plugin_manager;

    // Create a list of paths to plugins
    std::vector<std::filesystem::path> plugin_paths
        = {"./hipdnn_test_plugin1", "./hipdnn_test_plugin2"};

    // Load the plugins
    plugin_manager.load_plugins(plugin_paths);

    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 2); // Ensure two plugins are loaded

    // Check that the plugins have the correct names
    ASSERT_EQ(plugins[0].name(), "Plugin1");
    ASSERT_EQ(plugins[1].name(), "Plugin2");

    // Check that the plugins have the correct versions
    ASSERT_EQ(plugins[0].version(), "1.0");
    ASSERT_EQ(plugins[1].version(), "2.0");

    // Check that the plugins have the correct types
    ASSERT_EQ(plugins[0].type(), hipdnnPluginTypeUnspecified);
    ASSERT_EQ(plugins[1].type(), hipdnnPluginTypeUnspecified);
}
