// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <set>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <utility>

#include "plugin/plugin_core.hpp"

using namespace hipdnn_backend;

namespace
{

class Plugin : public plugin::Plugin_base
{
public:
    // Forward constructor to base class
    explicit Plugin(plugin::Shared_library&& lib)
        : Plugin_base(std::move(lib))
    {
    }

    static hipdnnPluginType_t get_plugin_type()
    {
        return HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    }

    using Plugin_base::_lib;
    using Plugin_base::get_last_error_string;

private:
    friend class plugin::Plugin_manager_base<Plugin>;
};

class Test_plugin_manager : public plugin::Plugin_manager_base<Plugin>
{
public:
    Test_plugin_manager()
        : plugin::Plugin_manager_base<Plugin>({"test_plugins_dir"})
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

#if defined(_WIN32)
constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";
#else
constexpr const char* SHARED_LIB_EXT = ".so";
constexpr const char* LIB_PREFIX = "lib";
#endif

const std::string PLUGIN_NAME1 = "hipdnn_test_plugin1";
const std::string PLUGIN_NAME2 = "hipdnn_test_plugin2";

const std::string PLUGIN_PATH1 = "./" + PLUGIN_NAME1;
const std::string PLUGIN_PATH2 = "./" + PLUGIN_NAME2;

const std::string FULL_PLUGIN_PATH1
    = std::string("./") + LIB_PREFIX + PLUGIN_NAME1 + SHARED_LIB_EXT;
const std::string FULL_PLUGIN_PATH2
    = std::string("./") + LIB_PREFIX + PLUGIN_NAME2 + SHARED_LIB_EXT;

} // namespace

TEST(PluginManagerTest, LoadPlugins)
{
    // Create a PluginManager instance
    Test_plugin_manager plugin_manager;

    // Create a list of paths to plugins
    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1, PLUGIN_PATH2};

    // Load the plugins
    plugin_manager.load_plugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

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

TEST(PluginManagerTest, LoadPluginsFromDirectory)
{
    const std::filesystem::path temp_dir = "./temp_plugin_dir";
    std::filesystem::create_directory(temp_dir);

    try
    {
        std::filesystem::copy_file(FULL_PLUGIN_PATH1,
                                   temp_dir / std::filesystem::path(FULL_PLUGIN_PATH1).filename());
        std::filesystem::copy_file(FULL_PLUGIN_PATH2,
                                   temp_dir / std::filesystem::path(FULL_PLUGIN_PATH2).filename());

        Test_plugin_manager plugin_manager;
        plugin_manager.load_plugins({temp_dir}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 2);

        std::set<std::string_view> plugin_names;
        for(const auto& p : plugins)
        {
            plugin_names.insert(p.name());
        }
        EXPECT_TRUE(plugin_names.contains("Plugin1"));
        EXPECT_TRUE(plugin_names.contains("Plugin2"));
    }
    catch(...)
    {
        std::filesystem::remove_all(temp_dir);
        FAIL();
    }
    std::filesystem::remove_all(temp_dir);
}

TEST(PluginManagerTest, LoadPluginsAbsolute)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.load_plugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(plugin_manager.get_plugins().size(), 1);

    plugin_manager.load_plugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 1);
    EXPECT_EQ(plugins[0].name(), "Plugin2");
}

TEST(PluginManagerTest, LoadPluginsAdditive)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.load_plugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(plugin_manager.get_plugins().size(), 1);

    plugin_manager.load_plugins({PLUGIN_PATH1, PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    const auto& plugins = plugin_manager.get_plugins();
    EXPECT_EQ(plugins.size(), 2);
}

TEST(PluginManagerTest, LoadPlugins_AdditiveAccumulates)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.load_plugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(plugin_manager.get_plugins().size(), 1);

    plugin_manager.load_plugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 2);

    std::set<std::string_view> plugin_names;
    for(const auto& p : plugins)
    {
        plugin_names.insert(p.name());
    }
    EXPECT_TRUE(plugin_names.contains("Plugin1"));
    EXPECT_TRUE(plugin_names.contains("Plugin2"));
}

TEST(PluginManagerTest, LoadPlugins_AbsoluteReplaces)
{
    Test_plugin_manager plugin_manager;
    plugin_manager.load_plugins({PLUGIN_PATH1}, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(plugin_manager.get_plugins().size(), 1);

    plugin_manager.load_plugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 1);
    EXPECT_EQ(plugins[0].name(), "Plugin2");
}

TEST(PluginManagerTest, LoadPluginsAdditiveWithDefault)
{
    const std::filesystem::path default_dir = "test_plugins_dir";
    std::filesystem::create_directory(default_dir);

    try
    {
        // Place a plugin in the default directory
        std::filesystem::copy_file(
            FULL_PLUGIN_PATH1, default_dir / std::filesystem::path(FULL_PLUGIN_PATH1).filename());

        Test_plugin_manager plugin_manager;
        plugin_manager.load_plugins({PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ADDITIVE);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 2);

        // Verify both plugins (default and custom) were loaded
        std::set<std::string_view> plugin_names;
        for(const auto& p : plugins)
        {
            plugin_names.insert(p.name());
        }
        EXPECT_TRUE(plugin_names.contains("Plugin1"));
        EXPECT_TRUE(plugin_names.contains("Plugin2"));
    }
    catch(...)
    {
        std::filesystem::remove_all(default_dir);
        FAIL();
    }
    std::filesystem::remove_all(default_dir);
}

TEST(PluginManagerTest, LoadPluginsCombinedFileAndDirectory)
{
    const std::filesystem::path temp_dir = "./temp_plugin_dir_combined";
    std::filesystem::create_directory(temp_dir);

    try
    {
        std::filesystem::copy_file(FULL_PLUGIN_PATH1,
                                   temp_dir / std::filesystem::path(FULL_PLUGIN_PATH1).filename());

        Test_plugin_manager plugin_manager;
        plugin_manager.load_plugins({temp_dir, PLUGIN_PATH2}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 2);

        std::set<std::string_view> plugin_names;
        for(const auto& p : plugins)
        {
            plugin_names.insert(p.name());
        }
        EXPECT_TRUE(plugin_names.contains("Plugin1"));
        EXPECT_TRUE(plugin_names.contains("Plugin2"));
    }
    catch(...)
    {
        std::filesystem::remove_all(temp_dir);
        FAIL();
    }
    std::filesystem::remove_all(temp_dir);
}

TEST(PluginManagerTest, LastError)
{
    Test_plugin_manager plugin_manager;

    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1};
    plugin_manager.load_plugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = plugin_manager.get_plugins();
    ASSERT_EQ(plugins.size(), 1);

    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = plugins[0]._lib.get_symbol<Func_type>("hipdnnPluginGetName");

    ASSERT_TRUE(plugins[0].get_last_error_string().empty());

    ASSERT_NE(func_get_name(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(plugins[0].get_last_error_string(), "name is null");
}

TEST(PluginManagerTest, LastErrorMultithreaded)
{
    Test_plugin_manager plugin_manager;

    std::set<std::filesystem::path> plugin_paths = {PLUGIN_PATH1};
    plugin_manager.load_plugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

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
               && plugins[0].get_last_error_string() == "name is null";
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
        plugin_manager.load_plugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 1);

        auto func_get_name = plugins[0]._lib.get_symbol<Func_type>(func_name);
        func_get_name(nullptr);
    }

    {
        Test_plugin_manager plugin_manager;
        plugin_manager.load_plugins(plugin_paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

        const auto& plugins = plugin_manager.get_plugins();
        ASSERT_EQ(plugins.size(), 1);

        auto func_get_name = plugins[0]._lib.get_symbol<Func_type>(func_name);

        ASSERT_TRUE(plugins[0].get_last_error_string().empty());
        ASSERT_NE(func_get_name(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_EQ(plugins[0].get_last_error_string(), "name is null");
    }
}

TEST(PluginTest, SetLoggingCallback)
{
    g_callback_was_called = false;

    plugin::Shared_library lib(PLUGIN_PATH1);

    Plugin plugin(std::move(lib));

    EXPECT_EQ(plugin.set_logging_callback(dummy_callback), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_TRUE(g_callback_was_called);
    EXPECT_EQ(plugin.set_logging_callback(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}
