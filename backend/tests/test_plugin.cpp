// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <thread>

#include <gtest/gtest.h>

#include "plugin/plugin_core.hpp"

using namespace hipdnn_backend;

namespace
{

class Plugin : public plugin::Plugin_base
{
public:
    using Plugin_base::_lib;
    using Plugin_base::get_last_error_string;

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
    ASSERT_EQ(plugins[0].type(), HIPDNN_PLUGIN_TYPE_UNSPECIFIED);
    ASSERT_EQ(plugins[1].type(), HIPDNN_PLUGIN_TYPE_UNSPECIFIED);
}

TEST(PluginManagerTest, LastError)
{
    plugin::Plugin_manager_base<Plugin> plugin_manager;

    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_plugin1"};
    plugin_manager.load_plugins(plugin_paths);

    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 1);

    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = plugins[0]._lib.get_symbol<Func_type>("hipdnnPluginGetName");

    ASSERT_TRUE(plugins[0].get_last_error_string().empty());

    ASSERT_NE(func_get_name(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(plugins[0].get_last_error_string(), "hipdnnPluginGetName: name is null");
}

TEST(PluginManagerTest, LastErrorMultithreaded)
{
    plugin::Plugin_manager_base<Plugin> plugin_manager;

    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_plugin1"};
    plugin_manager.load_plugins(plugin_paths);

    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 1);

    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = plugins[0]._lib.get_symbol<Func_type>("hipdnnPluginGetName");

    auto check_get_name = [&]() {
        if(!plugins[0].get_last_error_string().empty())
        {
            return false;
        }

        return func_get_name(nullptr) != HIPDNN_PLUGIN_STATUS_SUCCESS
               && plugins[0].get_last_error_string() == "hipdnnPluginGetName: name is null";
    };

    ASSERT_EQ(check_get_name(), true);

    bool result1 = false;
    bool result2 = false;

    std::thread thread1([&] { result1 = check_get_name(); });

    std::thread thread2([&] { result2 = check_get_name(); });

    thread1.join();
    thread2.join();

    ASSERT_EQ(result1, true);
    ASSERT_EQ(result2, true);
}

TEST(PluginManagerTest, LastErrorOnSecondLoad)
{
    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    const auto func_name = "hipdnnPluginGetName";

    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_plugin1"};

    {
        plugin::Plugin_manager_base<Plugin> plugin_manager;
        plugin_manager.load_plugins(plugin_paths);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 1);

        auto func_get_name = plugins[0]._lib.get_symbol<Func_type>(func_name);
        func_get_name(nullptr);
    }

    {
        plugin::Plugin_manager_base<Plugin> plugin_manager;
        plugin_manager.load_plugins(plugin_paths);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 1);

        auto func_get_name = plugins[0]._lib.get_symbol<Func_type>(func_name);

        ASSERT_TRUE(plugins[0].get_last_error_string().empty());
        ASSERT_NE(func_get_name(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_EQ(plugins[0].get_last_error_string(), "hipdnnPluginGetName: name is null");
    }
}
