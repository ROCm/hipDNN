// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/plugin_api_enums.h>

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
    ASSERT_EQ(status, hipdnnPluginStatusSuccess);
    ASSERT_NE(name, nullptr);
    ASSERT_STREQ(name, "Plugin1");
}
