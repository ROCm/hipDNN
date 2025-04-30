// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plugin/shared_library.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/plugin_api_enums.h>

using namespace hipdnn_backend;

namespace
{

const char* const LIBRARY_PATH       = "./hipdnn_test_plugin1";
const char* const WRONG_LIBRARY_PATH = "./wrong_path";
const char* const SYMBOL_NAME        = "hipdnnPluginGetName";
const char* const WRONG_SYMBOL_NAME  = "wrong_symbol_name";

}

TEST(SharedLibraryTest, LoadLibrary)
{
    plugin::Shared_library library;
    ASSERT_TRUE(library.load(LIBRARY_PATH));
    library.unload();
}

TEST(SharedLibraryTest, LoadLibraryCtor)
{
    plugin::Shared_library library(LIBRARY_PATH);
}

TEST(SharedLibraryTest, LoadLibraryWrongPath)
{
    plugin::Shared_library library;
    ASSERT_FALSE(library.load(WRONG_LIBRARY_PATH));
    library.unload();
}

TEST(SharedLibraryTest, LoadLibraryCtorWrongPath)
{
    ASSERT_THROW(plugin::Shared_library library(WRONG_LIBRARY_PATH),
                 std::runtime_error // TODO Replace with the actual exception type thrown
    );
}

TEST(SharedLibraryTest, GetSymbol)
{
    plugin::Shared_library library(LIBRARY_PATH);

    auto symbol = library.get_symbol(SYMBOL_NAME);
    ASSERT_NE(symbol, nullptr);
}

TEST(SharedLibraryTest, GetSymbolUninitialized)
{
    plugin::Shared_library library;
    ASSERT_THROW(library.get_symbol(SYMBOL_NAME),
                 std::runtime_error // TODO Replace with the actual exception type thrown
    );
}

TEST(SharedLibraryTest, GetSymbolWrongName)
{
    plugin::Shared_library library(LIBRARY_PATH);

    auto symbol = library.get_symbol(WRONG_SYMBOL_NAME);
    ASSERT_EQ(symbol, nullptr);
}

TEST(SharedLibraryTest, CallFunction)
{
    plugin::Shared_library library(LIBRARY_PATH);

    // Get the function pointer
    using Func_type    = hipdnnPluginStatus_t (*)(const char**);
    auto func_get_name = library.get_symbol<Func_type>(SYMBOL_NAME);
    ASSERT_NE(func_get_name, nullptr);

    // Call the function to get the plugin name
    const char* name   = nullptr;
    auto        status = func_get_name(&name);
    ASSERT_EQ(status, hipdnnPluginStatusSuccess);
    ASSERT_NE(name, nullptr);
    ASSERT_STREQ(name, "Plugin1");
}
