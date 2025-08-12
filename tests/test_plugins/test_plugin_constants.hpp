// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

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

inline const char* test_good_plugin_name = TEST_GOOD_PLUGIN_NAME;
inline const char* test_execute_fails_plugin_name = TEST_EXECUTE_FAILS_PLUGIN_NAME;
inline const char* test_no_applicable_engines_plugin_name = TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME;

// Compose full plugin path with existence checking
inline std::string get_plugin_path(const char* plugin_name)
{
    namespace fs = std::filesystem;

    fs::path plugin_file = PLUGIN_DIR / hipdnn_sdk::utilities::get_library_name(plugin_name);

    // Check if the file exists
    if(!fs::exists(plugin_file))
    {
        throw std::runtime_error("Plugin file not found: " + plugin_file.string());
    }

    return plugin_file.string();
}

inline const std::string& test_good_plugin_path()
{
    static const std::string test_good_plugin_path = get_plugin_path(test_good_plugin_name);
    return test_good_plugin_path;
}

inline const std::string& test_execute_fails_plugin_path()
{
    static const std::string test_execute_fails_plugin_path
        = get_plugin_path(test_execute_fails_plugin_name);
    return test_execute_fails_plugin_path;
}

inline const std::string& test_no_applicable_engines_plugin_path()
{
    static const std::string test_no_applicable_engines_plugin_path
        = get_plugin_path(test_no_applicable_engines_plugin_name);
    return test_no_applicable_engines_plugin_path;
}

} // namespace plugin_constants
} // namespace hipdnn_tests
