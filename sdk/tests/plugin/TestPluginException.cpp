// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/PluginException.hpp>

using namespace hipdnn_plugin;

TEST(HipdnnPluginExceptionTest, WhatReturnsMessage)
{
    HipdnnPluginException ex(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "bad param");
    EXPECT_STREQ(ex.what(), "bad param");
    EXPECT_EQ(ex.getMessage(), std::string("bad param"));
    EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(HipdnnPluginExceptionTest, ThrowIfNeMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_NE(1, 2, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "not equal"),
                 HipdnnPluginException);
}

TEST(HipdnnPluginExceptionTest, ThrowIfEqMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_EQ(1, 1, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "equal"),
                 HipdnnPluginException);
}

TEST(HipdnnPluginExceptionTest, ThrowIfTrueMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_TRUE(true, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "true"),
                 HipdnnPluginException);
}

TEST(HipdnnPluginExceptionTest, ThrowIfFalseMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_FALSE(false, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "false"),
                 HipdnnPluginException);
}

TEST(HipdnnPluginExceptionTest, ThrowIfNullMacroThrows)
{
    void* ptr = nullptr;
    EXPECT_THROW(PLUGIN_THROW_IF_NULL(ptr, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "null"),
                 HipdnnPluginException);
}

TEST(HipdnnPluginExceptionTest, ThrowIfLtMacroThrows)
{
    EXPECT_THROW(PLUGIN_THROW_IF_LT(1, 2, HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "less than"),
                 HipdnnPluginException);
}
