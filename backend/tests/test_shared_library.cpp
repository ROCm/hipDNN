// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "descriptors/test_macros.hpp"
#include "hipdnn_exception.hpp"
#include "plugin/shared_library.hpp"

using namespace hipdnn_backend;

namespace
{

const char* const LIBRARY_PATH = "./hipdnn_test_plugin1";
const char* const WRONG_LIBRARY_PATH = "./wrong_path";
const char* const SYMBOL_NAME = "hipdnnPluginGetName";
const char* const WRONG_SYMBOL_NAME = "wrong_symbol_name";

#if defined(_WIN32)
const char* const FULL_LIBRARY_PATH = "./hipdnn_test_plugin1.dll";
#else
const char* const FULL_LIBRARY_PATH = "./libhipdnn_test_plugin1.so";
#endif

}

TEST(SharedLibraryTest, LoadLibrary)
{
    plugin::Shared_library library;
    library.load(LIBRARY_PATH);
    library.unload();
}

TEST(SharedLibraryTest, LoadLibraryCtor)
{
    plugin::Shared_library library(LIBRARY_PATH);
}

TEST(SharedLibraryTest, LoadLibraryWrongPath)
{
    plugin::Shared_library library;
    ASSERT_THROW_HIPDNN_STATUS(library.load(WRONG_LIBRARY_PATH), HIPDNN_STATUS_PLUGIN_ERROR);
    library.unload();
}

TEST(SharedLibraryTest, LoadLibraryCtorWrongPath)
{
    ASSERT_THROW_HIPDNN_STATUS(plugin::Shared_library(WRONG_LIBRARY_PATH),
                               HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST(SharedLibraryTest, GetSymbol)
{
    plugin::Shared_library library(LIBRARY_PATH);

    ASSERT_NO_THROW(library.get_symbol(SYMBOL_NAME));
}

TEST(SharedLibraryTest, GetSymbolUninitialized)
{
    plugin::Shared_library library;
    ASSERT_THROW_HIPDNN_STATUS(library.get_symbol(SYMBOL_NAME), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST(SharedLibraryTest, GetSymbolWrongName)
{
    plugin::Shared_library library(LIBRARY_PATH);

    ASSERT_THROW_HIPDNN_STATUS(library.get_symbol(WRONG_SYMBOL_NAME), HIPDNN_STATUS_PLUGIN_ERROR);
}

TEST(SharedLibraryTest, CallFunction)
{
    plugin::Shared_library library(LIBRARY_PATH);

    // Get the function pointer
    using Func_type = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = library.get_symbol<Func_type>(SYMBOL_NAME);

    // Call the function to get the plugin name
    const char* name = nullptr;
    auto status = func_get_name(&name);
    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(name, nullptr);
    ASSERT_STREQ(name, "Plugin1");
}

TEST(Shared_Library_Test, get_current_module_directory_from_executable)
{
    std::filesystem::path path;
    ASSERT_NO_THROW(path = plugin::Shared_library::get_current_module_directory());

    EXPECT_FALSE(path.empty());
    EXPECT_TRUE(path.is_absolute());
    EXPECT_TRUE(std::filesystem::is_directory(path));

    // Only tests that it works from a statically linked binary
    EXPECT_TRUE(std::filesystem::exists(path / "hipdnn_backend_tests"));
}

class Shared_library_path_test : public ::testing::TestWithParam<std::string>
{
};

TEST_P(Shared_library_path_test, LoadWithValidPathFormats)
{
    const auto& path_param = GetParam();
    plugin::Shared_library library;
    ASSERT_NO_THROW(library.load(path_param));
}

INSTANTIATE_TEST_SUITE_P(PathVariations,
                         Shared_library_path_test,
                         ::testing::Values(
                             // Path without extension
                             std::string(LIBRARY_PATH),
                             // Path with full filename
                             std::string(FULL_LIBRARY_PATH),
                             // Absolute path
                             std::filesystem::absolute(FULL_LIBRARY_PATH).string()));
