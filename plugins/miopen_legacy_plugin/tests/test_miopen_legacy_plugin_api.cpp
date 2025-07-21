// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <gtest/gtest.h>

#include "hipdnn_sdk/plugin/plugin_api.h"

TEST(MiopenLegacyPluginApiTest, GetNameSuccess)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetName(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "miopen_legacy_plugin");
}

TEST(MiopenLegacyPluginApiTest, GetNameNullptr)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyPluginApiTest, GetVersionSuccess)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, "1.0.0");
}

TEST(MiopenLegacyPluginApiTest, GetVersionNullptr)
{
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyPluginApiTest, GetTypeSuccess)
{
    hipdnnPluginType_t type;
    EXPECT_EQ(hipdnnPluginGetType(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST(MiopenLegacyPluginApiTest, GetTypeNullptr)
{
    EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(MiopenLegacyPluginApiTest, GetLastErrorStringSuccess)
{
    const char* error_str = nullptr;
    hipdnnPluginGetLastErrorString(&error_str);
    ASSERT_NE(error_str, nullptr);
    EXPECT_GE(strlen(error_str), 0u);
}

TEST(MiopenLegacyPluginApiTest, GetLastErrorStringNullptr)
{
    EXPECT_NO_THROW(hipdnnPluginGetLastErrorString(nullptr));
}
