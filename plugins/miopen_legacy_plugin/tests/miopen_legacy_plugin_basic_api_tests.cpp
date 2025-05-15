// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_sdk/plugin/plugin_api.h"
#include <array> // line 5, miopen_legacy_plugin_basic_api_tests.cpp
#include <gtest/gtest.h>

// Test hipdnnPluginGetName
TEST(MiopenLegacyPluginApiTest, GetNameSuccess)
{
    const char* name = nullptr;
    EXPECT_EQ(hipdnnPluginGetName(&name), hipdnnPluginStatusSuccess);
    EXPECT_STREQ(name, "miopen_legacy_plugin");
}

TEST(MiopenLegacyPluginApiTest, GetNameNullptr)
{
    EXPECT_EQ(hipdnnPluginGetName(nullptr), hipdnnPluginStatusBadParam);
}

// Test hipdnnPluginGetVersion
TEST(MiopenLegacyPluginApiTest, GetVersionSuccess)
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetVersion(&version), hipdnnPluginStatusSuccess);
    EXPECT_STREQ(version, "1.0.0");
}

TEST(MiopenLegacyPluginApiTest, GetVersionNullptr)
{
    EXPECT_EQ(hipdnnPluginGetVersion(nullptr), hipdnnPluginStatusBadParam);
}

// Test hipdnnPluginGetType
TEST(MiopenLegacyPluginApiTest, GetTypeSuccess)
{
    hipdnnPluginType_t type;
    EXPECT_EQ(hipdnnPluginGetType(&type), hipdnnPluginStatusSuccess);
    EXPECT_EQ(type, hipdnnPluginTypeEngine);
}

TEST(MiopenLegacyPluginApiTest, GetTypeNullptr)
{
    EXPECT_EQ(hipdnnPluginGetType(nullptr), hipdnnPluginStatusBadParam);
}

// Test hipdnnPluginGetNumEngines
TEST(MiopenLegacyPluginApiTest, GetNumEnginesSuccess)
{
    unsigned num_engines = 0;
    EXPECT_EQ(hipdnnPluginGetNumEngines(&num_engines), hipdnnPluginStatusSuccess);
    EXPECT_EQ(num_engines, 1u);
}

TEST(MiopenLegacyPluginApiTest, GetNumEnginesNullptr)
{
    EXPECT_EQ(hipdnnPluginGetNumEngines(nullptr), hipdnnPluginStatusBadParam);
}

// Test hipdnnPluginRunEngine
TEST(MiopenLegacyPluginApiTest, RunEngineSuccess)
{
    std::array<uint32_t, 4> input
        = {1, 2, 3, 4}; // line 54, miopen_legacy_plugin_basic_api_tests.cpp
    std::array<uint32_t, 4> output = {0}; // line 55, miopen_legacy_plugin_basic_api_tests.cpp
    EXPECT_EQ(hipdnnPluginRunEngine(0, input.data(), output.data(), input.size()),
              hipdnnPluginStatusSuccess);
    for(size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_EQ(output[i], input[i]);
    }
}

TEST(MiopenLegacyPluginApiTest, RunEngineBadParam)
{
    std::array<uint32_t, 1> dummy = {0}; // line 64, miopen_legacy_plugin_basic_api_tests.cpp
    // Bad engine index
    EXPECT_EQ(hipdnnPluginRunEngine(1, dummy.data(), dummy.data(), dummy.size()),
              hipdnnPluginStatusBadParam);
    // Null input
    EXPECT_EQ(hipdnnPluginRunEngine(0, nullptr, dummy.data(), dummy.size()),
              hipdnnPluginStatusBadParam);
    // Null output
    EXPECT_EQ(hipdnnPluginRunEngine(0, dummy.data(), nullptr, dummy.size()),
              hipdnnPluginStatusBadParam);
}
