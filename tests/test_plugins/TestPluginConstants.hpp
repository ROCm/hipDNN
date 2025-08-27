// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "TestPluginEngineIdMap.hpp"
#include <filesystem>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <stdexcept>
#include <string>

namespace hipdnn_tests
{
namespace plugin_constants
{
// Test plugin directory relative to test executables
const std::filesystem::path PLUGIN_DIR = "../test_plugins";

// Compose full plugin path with existence checking
inline std::string getPluginPath(const char* pluginName)
{
    namespace fs = std::filesystem;

    fs::path pluginFile = PLUGIN_DIR / hipdnn_sdk::utilities::getLibraryName(pluginName);

    // Check if the file exists
    if(!fs::exists(pluginFile))
    {
        throw std::runtime_error("Plugin file not found: " + pluginFile.string());
    }

    return pluginFile.string();
}

inline const std::string& testGoodPluginPath()
{
    static const std::string s_testGoodPluginPath = getPluginPath(TEST_GOOD_PLUGIN_NAME);
    return s_testGoodPluginPath;
}

inline const std::string& testExecuteFailsPluginPath()
{
    static const std::string s_testExecuteFailsPluginPath
        = getPluginPath(TEST_EXECUTE_FAILS_PLUGIN_NAME);
    return s_testExecuteFailsPluginPath;
}

inline const std::string& testNoApplicableEnginesPluginPath()
{
    static const std::string s_testNoApplicableEnginesPluginPath
        = getPluginPath(TEST_NO_APPLICABLE_ENGINES_PLUGIN_NAME);
    return s_testNoApplicableEnginesPluginPath;
}

inline const std::string& testDuplicateIdAPluginPath()
{
    static const std::string s_testDuplicateIdAPluginPath
        = getPluginPath(TEST_DUPLICATE_ID_A_PLUGIN_NAME);
    return s_testDuplicateIdAPluginPath;
}

inline const std::string& testDuplicateIdBPluginPath()
{
    static const std::string s_testDuplicateIdBPluginPath
        = getPluginPath(TEST_DUPLICATE_ID_B_PLUGIN_NAME);
    return s_testDuplicateIdBPluginPath;
}

inline const std::string& testIncompleteApiPluginPath()
{
    static const std::string s_testIncompleteApiPluginPath
        = getPluginPath(TEST_INCOMPLETE_API_PLUGIN_NAME);
    return s_testIncompleteApiPluginPath;
}

} // namespace plugin_constants
} // namespace hipdnn_tests
