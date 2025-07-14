// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

using namespace hipdnn_plugin;

TEST(PluginHelpersTest, TryCatchReturnsSuccessOnNoException)
{
    auto lambda = []() {};
    auto status = try_catch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(PluginHelpersTest, TryCatchHandlesHipdnnException)
{
    auto lambda = []() {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "internal error");
    };
    auto status = try_catch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), "internal error");
}

TEST(PluginHelpersTest, TryCatchHandlesStdException)
{
    auto lambda = []() { throw std::runtime_error("std exception"); };
    auto status = try_catch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), "std exception");
}

TEST(PluginHelpersTest, TryCatchHandlesUnknownException)
{
    auto lambda = []() { throw 42; };
    auto status = try_catch(lambda);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), "Unknown exception occured");
}
