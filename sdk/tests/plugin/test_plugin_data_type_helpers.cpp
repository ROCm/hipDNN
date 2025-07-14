// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <sstream>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>

TEST(PluginDataTypeHelpersTest, ToStringPluginStatus)
{
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_STATUS_SUCCESS), "HIPDNN_PLUGIN_STATUS_SUCCESS");
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_STATUS_BAD_PARAM), "HIPDNN_PLUGIN_STATUS_BAD_PARAM");
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_STATUS_INVALID_VALUE),
                 "HIPDNN_PLUGIN_STATUS_INVALID_VALUE");
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR),
                 "HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR");
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_STATUS_ALLOC_FAILED), "HIPDNN_PLUGIN_STATUS_ALLOC_FAILED");

    EXPECT_STREQ(to_string(static_cast<hipdnnPluginStatus_t>(999)), "HIPDNN_PLUGIN_STATUS_UNKNOWN");
}

TEST(PluginDataTypeHelpersTest, ToStringPluginType)
{
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_TYPE_UNSPECIFIED), "HIPDNN_PLUGIN_TYPE_UNSPECIFIED");
    EXPECT_STREQ(to_string(HIPDNN_PLUGIN_TYPE_ENGINE), "HIPDNN_PLUGIN_TYPE_ENGINE");
    EXPECT_STREQ(to_string(static_cast<hipdnnPluginType_t>(999)), "HIPDNN_PLUGIN_TYPE_UNKNOWN");
}

TEST(PluginDataTypeHelpersTest, OstreamOperatorPluginStatus)
{
    std::ostringstream oss;
    oss << HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    EXPECT_EQ(oss.str(), "HIPDNN_PLUGIN_STATUS_BAD_PARAM");
}

TEST(PluginDataTypeHelpersTest, OstreamOperatorPluginType)
{
    std::ostringstream oss;
    oss << HIPDNN_PLUGIN_TYPE_ENGINE;
    EXPECT_EQ(oss.str(), "HIPDNN_PLUGIN_TYPE_ENGINE");
}
