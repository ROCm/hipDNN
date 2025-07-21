// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

using namespace hipdnn_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

TEST(PluginLastErrorManagerTest, SetAndGetLastErrorString)
{
    const char* msg = "test error message";
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, msg);
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
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, prev_msg);
    Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_SUCCESS, "should not overwrite");
    EXPECT_STREQ(Plugin_last_error_manager::get_last_error(), prev_msg);
}
