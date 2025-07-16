// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

#include "plugin/engine_plugin_manager.hpp"

using namespace hipdnn_backend;

TEST(GPU_EnginePluginManagerTest, LoadPluginsAndExecuteOpGraph)
{
    SKIP_IF_NO_DEVICES();

    // Create an EngienPluginManager instance
    plugin::Engine_plugin_manager plugin_manager;

    // Create a list of paths to plugins
    std::vector<std::filesystem::path> plugin_paths = {"./hipdnn_test_engine_plugin1"};

    // Load the plugins
    plugin_manager.load_plugins(plugin_paths);
}
