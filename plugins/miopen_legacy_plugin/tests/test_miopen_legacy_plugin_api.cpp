// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_sdk/plugin/plugin_api.h"
#include <array>
#include <gtest/gtest.h>
#include <iostream>

namespace
{
void test_logging_callback(hipdnnSeverity_t severity, const char* msg)
{
    (void)severity;
    // std::cout << msg << "\n"; // uncomment to see formatted log messages during tests.
    // It does not use the true callback yet since the plugin is not yet loaded.
    (void)msg;
}

class Miopen_legacy_plugin_api_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnPluginSetLoggingCallback(test_logging_callback),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
};
} // namespace

TEST_F(Miopen_legacy_plugin_api_test, GetNameSuccess)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetName(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "miopen_legacy_plugin");
}

TEST_F(Miopen_legacy_plugin_api_test, GetNameNullptr)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(Miopen_legacy_plugin_api_test, GetVersionSuccess)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_STREQ(version, "1.0.0");
}

TEST_F(Miopen_legacy_plugin_api_test, GetVersionNullptr)
{
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(Miopen_legacy_plugin_api_test, GetTypeSuccess)
{
    hipdnnPluginType_t type;
    EXPECT_EQ(hipdnnPluginGetType(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST_F(Miopen_legacy_plugin_api_test, GetTypeNullptr)
{
    EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(Miopen_legacy_plugin_api_test, GetLastErrorStringSuccess)
{
    const char* error_str = nullptr;
    hipdnnPluginGetLastErrorString(&error_str);
    ASSERT_NE(error_str, nullptr);
    EXPECT_GE(strlen(error_str), 0u);
}

TEST_F(Miopen_legacy_plugin_api_test, GetLastErrorStringNullptr)
{
    EXPECT_NO_THROW(hipdnnPluginGetLastErrorString(nullptr));
}
