// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "test_plugin_engine_id_map.hpp"
#include <filesystem>
#include <hipdnn_sdk/utilities/platform_utils.hpp>
#include <stdexcept>
#include <string>

namespace hipdnn_tests
{
namespace plugin_constants
{
// Test plugin directory relative to test executables
const std::filesystem::path PLUGIN_DIR = "../test_plugins";

// Compose full plugin path with existence checking
inline std::string get_plugin_path(const char* plugin_name)
{
    namespace fs = std::filesystem;

    fs::path plugin_file = PLUGIN_DIR / hipdnn_sdk::utilities::getLibraryName(plugin_name);

    // Check if the file exists
    if(!fs::exists(plugin_file))
    {
        throw std::runtime_error("Plugin file not found: " + plugin_file.string());
    }

    return plugin_file.string();
}

inline const std::string& testGoodPluginPath()
{
    static const std::string testGoodPluginPath = get_plugin_path(TEST_GOOD_PLUGIN_NAME);
    return testGoodPluginPath;
}

inline const std::string& testExecuteFailsPluginPath()
{
    static const std::string testExecuteFailsPluginPath
        = get_plugin_path(TEST_EXECUTE_FAILS_PLUGIN_NAME);
    return testExecuteFailsPluginPath;
}

inline const std::string& testNoApplicableEnginesPluginPath()
{
    static const std::string testNoApplicableEnginesPluginPath
        = get_plugin_path(TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME);
    return testNoApplicableEnginesPluginPath;
}

inline const std::string& testDuplicateIdAPluginPath()
{
    static const std::string testDuplicateIdAPluginPath
        = get_plugin_path(TEST_DUPLICATE_ID_A_PLUGIN_NAME);
    return testDuplicateIdAPluginPath;
}

inline const std::string& testDuplicateIdBPluginPath()
{
    static const std::string testDuplicateIdBPluginPath
        = get_plugin_path(TEST_DUPLICATE_ID_B_PLUGIN_NAME);
    return testDuplicateIdBPluginPath;
}

inline const std::string& testIncompleteApiPluginPath()
{
    static const std::string testIncompleteApiPluginPath
        = get_plugin_path(TEST_INCOMPLETE_API_PLUGIN_NAME);
    return testIncompleteApiPluginPath;
}

} // namespace plugin_constants
} // namespace hipdnn_tests
