// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/plugin_exception.hpp>

using namespace hipdnn_plugin;

TEST(HipdnnPluginExceptionTest, WhatReturnsMessage)
{
    Hipdnn_plugin_exception ex(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "bad param");
    EXPECT_STREQ(ex.what(), "bad param");
    EXPECT_EQ(ex.get_message(), std::string("bad param"));
    EXPECT_EQ(ex.get_status(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(HipdnnPluginExceptionTest, ThrowIfNeMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_NE(1, 2, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "not equal"),
                 Hipdnn_plugin_exception);
}

TEST(HipdnnPluginExceptionTest, ThrowIfEqMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_EQ(1, 1, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "equal"),
                 Hipdnn_plugin_exception);
}

TEST(HipdnnPluginExceptionTest, ThrowIfTrueMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_TRUE(true, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "true"),
                 Hipdnn_plugin_exception);
}

TEST(HipdnnPluginExceptionTest, ThrowIfFalseMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_FALSE(false, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "false"),
                 Hipdnn_plugin_exception);
}

TEST(HipdnnPluginExceptionTest, ThrowIfNullMacroThrows)
{
    void* ptr = nullptr;
    EXPECT_THROW(PLUGIN_THROW_IF_NULL(ptr, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "null"),
                 Hipdnn_plugin_exception);
}

TEST(HipdnnPluginExceptionTest, ThrowIfLtMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_LT(1, 2, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "less than"),
                 Hipdnn_plugin_exception);
}
