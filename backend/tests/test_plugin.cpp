// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <thread>

#include "platform_utils.hpp"
#include <gtest/gtest.h>
#include <utility>

#include "plugin/plugin_core.hpp"
#include <hipdnn_sdk/test_utilities/temp_directory.hpp>

using namespace hipdnn_backend;

namespace
{

class Plugin : public plugin::PluginBase
{
public:
    // Forward constructor to base class
    explicit Plugin(plugin::SharedLibrary&& lib)
        : PluginBase(std::move(lib))
    {
    }

    static hipdnnPluginType_t getPluginType()
    {
        return HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    }

    using PluginBase::_lib;
    using PluginBase::getLastErrorString;

private:
    friend class plugin::PluginManagerBase<Plugin>;
};

class Test_plugin_manager : public plugin::PluginManagerBase<Plugin>
{
public:
    Test_plugin_manager()
        : plugin::PluginManagerBase<Plugin>({"test_plugins_dir"})
    {
    }
    Test_plugin_manager(std::set<std::filesystem::path> paths)
        : plugin::PluginManagerBase<Plugin>(std::move(paths))
    {
    }
};

bool g_callback_was_called = false;
void dummy_callback(hipdnnSeverity_t sev, const char* msg)
{
    (void)sev;
    (void)msg;
    g_callback_was_called = true;
}

const std::string PLUGIN_NAME1 = "hipdnn_test_plugin1";
const std::string PLUGIN_NAME2 = "hipdnn_test_plugin2";

const std::filesystem::path PLUGIN_PATH1 = std::filesystem::path(".") /= PLUGIN_NAME1;
const std::filesystem::path PLUGIN_PATH2 = std::filesystem::path(".") /= PLUGIN_NAME2;

const std::filesystem::path FULL_PLUGIN_PATH1 = std::filesystem::path(".")
    /= hipdnn_sdk::utilities::get_library_name(PLUGIN_NAME1.c_str());
const std::filesystem::path FULL_PLUGIN_PATH2 = std::filesystem::path(".")
    /= hipdnn_sdk::utilities::get_library_name(PLUGIN_NAME2.c_str());

} // namespace

TEST(PluginManagerTest, LoadPlugins)
{
    // Create a PluginManager instance
    Test_plugin_manager plugin_manager;

    // Create a list of paths to plugins
    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1, PLUGIN_PATH2};

    // Load the plugins
    plugin_manager.loadPlugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 2); // Ensure two plugins are loaded

    // Check that the plugins have the correct names
    ASSERT_EQ(plugins[0]->name(), "Plugin1");
    ASSERT_EQ(plugins[1]->name(), "Plugin2");

    // Check that the plugins have the correct versions
    ASSERT_EQ(plugins[0]->version(), "1.0");
    ASSERT_EQ(plugins[1]->version(), "2.0");

    // Check that the plugins have the correct types
    ASSERT_EQ(plugins[0]->type(), HIPDNN_PLUGIN_TYPE_UNSPECIFIED);
    ASSERT_EQ(plugins[1]->type(), HIPDNN_PLUGIN_TYPE_UNSPECIFIED);
}

TEST(PluginManagerTest, LoadPluginsFromDirectory)
{
    Temp_directory temp_dir("temp_plugin_dir");

    std::filesystem::copy_file(
        FULL_PLUGIN_PATH1, temp_dir.path() / std::filesystem::path(FULL_PLUGIN_PATH1).filename());
    std::filesystem::copy_file(
        FULL_PLUGIN_PATH2, temp_dir.path() / std::filesystem::path(FULL_PLUGIN_PATH2).filename());

    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({temp_dir.path()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> plugin_names;
    for(const auto& p : plugins)
    {
        plugin_names.insert(p->name());
    }
    EXPECT_TRUE(plugin_names.contains("Plugin1"));
    EXPECT_TRUE(plugin_names.contains("Plugin2"));
}

TEST(PluginManagerTest, LoadPluginsAbsolute)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(plugin_manager.getPlugins().size(), 1);

    plugin_manager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);
    EXPECT_EQ(plugins[0]->name(), "Plugin2");
}

TEST(PluginManagerTest, LoadPluginsAdditive)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(plugin_manager.getPlugins().size(), 1);

    plugin_manager.loadPlugins({PLUGIN_PATH1, PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    const auto& plugins = plugin_manager.getPlugins();
    EXPECT_EQ(plugins.size(), 2);
}

TEST(PluginManagerTest, LoadPlugins_AdditiveAccumulates)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(plugin_manager.getPlugins().size(), 1);

    plugin_manager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> plugin_names;
    for(const auto& p : plugins)
    {
        plugin_names.insert(p->name());
    }
    EXPECT_TRUE(plugin_names.contains("Plugin1"));
    EXPECT_TRUE(plugin_names.contains("Plugin2"));
}

TEST(PluginManagerTest, LoadPlugins_AbsoluteReplaces)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(plugin_manager.getPlugins().size(), 1);

    plugin_manager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);
    EXPECT_EQ(plugins[0]->name(), "Plugin2");
}

TEST(PluginManagerTest, LoadPluginsAdditiveWithDefault)
{
    Temp_directory default_dir("test_plugins_dir");

    // Place a plugin in the default directory
    std::filesystem::copy_file(FULL_PLUGIN_PATH1,
                               default_dir.path()
                                   / std::filesystem::path(FULL_PLUGIN_PATH1).filename());

    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);

    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    // Verify both plugins (default and custom) were loaded
    std::set<std::string_view> plugin_names;
    for(const auto& p : plugins)
    {
        plugin_names.insert(p->name());
    }
    EXPECT_TRUE(plugin_names.contains("Plugin1"));
    EXPECT_TRUE(plugin_names.contains("Plugin2"));
}

TEST(PluginManagerTest, LoadPluginsCombinedFileAndDirectory)
{
    Temp_directory temp_dir("temp_plugin_dir_combined");

    std::filesystem::copy_file(
        FULL_PLUGIN_PATH1, temp_dir.path() / std::filesystem::path(FULL_PLUGIN_PATH1).filename());

    Test_plugin_manager plugin_manager;
    plugin_manager.loadPlugins({temp_dir.path(), PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> plugin_names;
    for(const auto& p : plugins)
    {
        plugin_names.insert(p->name());
    }
    EXPECT_TRUE(plugin_names.contains("Plugin1"));
    EXPECT_TRUE(plugin_names.contains("Plugin2"));
}

TEST(PluginManagerTest, LastError)
{
    Test_plugin_manager plugin_manager;

    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1};
    plugin_manager.loadPlugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);

    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = plugins[0]->_lib.get_symbol<Func_type>("hipdnnPluginGetName");

    ASSERT_TRUE(plugins[0]->getLastErrorString().empty());

    ASSERT_NE(func_get_name(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(plugins[0]->getLastErrorString(), "name is null");
}

TEST(PluginManagerTest, LastErrorMultithreaded)
{
    Test_plugin_manager plugin_manager;

    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1};
    plugin_manager.loadPlugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = plugin_manager.getPlugins();
    ASSERT_EQ(plugins.size(), 1);

    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = plugins[0]->_lib.get_symbol<Func_type>("hipdnnPluginGetName");

    auto check_get_name = [&]() {
        if(!plugins[0]->getLastErrorString().empty())
        {
            return false;
        }

        return func_get_name(nullptr) != HIPDNN_PLUGIN_STATUS_SUCCESS
               && plugins[0]->getLastErrorString() == "name is null";
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

    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1};

    {
        Test_plugin_manager plugin_manager;
        plugin_manager.loadPlugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = plugin_manager.getPlugins();
        ASSERT_EQ(plugins.size(), 1);

        auto func_get_name = plugins[0]->_lib.get_symbol<Func_type>(func_name);
        func_get_name(nullptr);
    }

    {
        Test_plugin_manager plugin_manager;
        plugin_manager.loadPlugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = plugin_manager.getPlugins();
        ASSERT_EQ(plugins.size(), 1);

        auto func_get_name = plugins[0]->_lib.get_symbol<Func_type>(func_name);

        ASSERT_TRUE(plugins[0]->getLastErrorString().empty());
        ASSERT_NE(func_get_name(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_EQ(plugins[0]->getLastErrorString(), "name is null");
    }
}

TEST(PluginTest, SetLoggingCallback)
{
    g_callback_was_called = false;

    plugin::SharedLibrary lib(PLUGIN_PATH1);

    Plugin plugin(std::move(lib));

    EXPECT_EQ(plugin.setLoggingCallback(dummy_callback), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(g_callback_was_called);
    EXPECT_EQ(plugin.setLoggingCallback(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}
