// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "descriptors/TestMacros.hpp"
#include "plugin/SharedLibrary.hpp"

using namespace hipdnn_backend;

namespace
{

const char* const LIBRARY_PATH = "./hipdnn_test_plugin1";
const char* const WRONG_LIBRARY_PATH = "./wrong_path";
const char* const SYMBOL_NAME = "hipdnnPluginGetName";
const char* const WRONG_SYMBOL_NAME = "wrong_symbol_name";

const std::string FULL_LIBRARY_PATH
    = (std::filesystem::path(".") /= hipdnn_sdk::utilities::getLibraryName("hipdnn_test_plugin1"))
          .string();

}

TEST(TestSharedLibrary, LoadLibrary)
{
    plugin::SharedLibrary library;
    library.load(LIBRARY_PATH);
    library.unload();
}

TEST(TestSharedLibrary, LoadLibraryCtor)
{
    plugin::SharedLibrary library(LIBRARY_PATH);
}

TEST(TestSharedLibrary, LoadLibraryWrongPath)
{
    plugin::SharedLibrary library;
    ASSERT_THROW_HIPDNN_STATUS(library.load(WRONG_LIBRARY_PATH), HIPDNN_STATUS_PLUGIN_ERROR);
    library.unload();
}

TEST(TestSharedLibrary, LoadLibraryCtorWrongPath)
{
    ASSERT_THROW_HIPDNN_STATUS(plugin::SharedLibrary(WRONG_LIBRARY_PATH),
                               HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST(TestSharedLibrary, GetSymbol)
{
    plugin::SharedLibrary library(LIBRARY_PATH);

    ASSERT_NO_THROW(library.getSymbol(SYMBOL_NAME));
}

TEST(TestSharedLibrary, GetSymbolUninitialized)
{
    plugin::SharedLibrary library;
    ASSERT_THROW_HIPDNN_STATUS(library.getSymbol(SYMBOL_NAME), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(TestSharedLibrary, GetSymbolWrongName)
{
    plugin::SharedLibrary library(LIBRARY_PATH);

    ASSERT_THROW_HIPDNN_STATUS(library.getSymbol(WRONG_SYMBOL_NAME), HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST(TestSharedLibrary, CallFunction)
{
    plugin::SharedLibrary library(LIBRARY_PATH);

    // Get the function pointer
    using FuncType = hipdnnPluginStatus_t (*)(const char**);
    auto funcGetName = library.getSymbol<FuncType>(SYMBOL_NAME);

    // Call the function to get the plugin name
    const char* name = nullptr;
    auto status = funcGetName(&name);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(name, nullptr);
    ASSERT_STREQ(name, "Plugin1");
}

TEST(TestSharedLibrary, GetCurrentModuleDirectoryFromExecutable)
{
    std::filesystem::path path;
    ASSERT_NO_THROW(path = platform_utilities::getCurrentModuleDirectory());

    EXPECT_FALSE(path.empty());
    EXPECT_TRUE(path.is_absolute());
    EXPECT_TRUE(std::filesystem::is_directory(path));

    // Only tests that it works from a statically linked binary
    EXPECT_TRUE(std::filesystem::exists(
        path / hipdnn_sdk::utilities::getExecutableName("hipdnn_backend_tests")));
}

class TestSharedLibraryPaths : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestSharedLibraryPaths, LoadWithValidPathFormats)
{
    const auto& pathParam = GetParam();
    plugin::SharedLibrary library;
    ASSERT_NO_THROW(library.load(pathParam));
}

INSTANTIATE_TEST_SUITE_P(PathVariations,
                         TestSharedLibraryPaths,
                         ::testing::Values(
                             // Path without extension
                             std::string(LIBRARY_PATH),
                             // Path with full filename
                             std::string(FULL_LIBRARY_PATH),
                             // Absolute path
                             std::filesystem::absolute(FULL_LIBRARY_PATH).string()));
