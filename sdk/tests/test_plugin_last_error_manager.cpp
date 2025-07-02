// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

using namespace hipdnn_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char Plugin_last_error_manager::last_error[HIPDNN_MAX_ERROR_STRING_SIZE] = "";

TEST(PluginLastErrorManagerTest, SetAndGetLastErrorString)
{
    const char* msg = "test error message";
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_INTERNAL_ERROR, msg);
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), msg);
}

TEST(PluginLastErrorManagerTest, SetLastErrorWithStdString)
{
    std::string msg = "std::string error";
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_BAD_PARAM, msg);
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), msg.c_str());
}

TEST(PluginLastErrorManagerTest, SetLastErrorSuccessDoesNotChangeError)
{
    const char* prev_msg = "previous error";
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_INTERNAL_ERROR, prev_msg);
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_SUCCESS, "should not overwrite");
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), prev_msg);
}

TEST(PluginLastErrorManagerTest, StatusStringMapping)
{
    EXPECT_STREQ(hipdnn_plugin_get_status_string(HIPDNN_PLUGIN_STATUS_SUCCESS),
                 "HIPDNN_PLUGIN_STATUS_SUCCESS");
    EXPECT_STREQ(hipdnn_plugin_get_status_string(HIPDNN_PLUGIN_STATUS_BAD_PARAM),
                 "HIPDNN_PLUGIN_STATUS_BAD_PARAM");
    EXPECT_STREQ(hipdnn_plugin_get_status_string(HIPDNN_PLUGIN_INVALID_VALUE),
                 "HIPDNN_PLUGIN_INVALID_VALUE");
    EXPECT_STREQ(hipdnn_plugin_get_status_string(HIPDNN_PLUGIN_INTERNAL_ERROR),
                 "HIPDNN_PLUGIN_INTERNAL_ERROR");
    EXPECT_STREQ(hipdnn_plugin_get_status_string(static_cast<hipdnnPluginStatus_t>(999)),
                 "HIPDNN_PLUGIN_STATUS_UNKNOWN");
}
