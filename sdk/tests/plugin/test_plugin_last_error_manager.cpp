// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#define HIPDNN_BACKEND_STATIC_DEFINE

#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

using namespace hipdnn_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char PluginLastErrorManager::_last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

TEST(PluginLastErrorManagerTest, SetAndGetLastErrorString)
{
    const char* msg = "test error message";
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, msg);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), msg);
}

TEST(PluginLastErrorManagerTest, SetLastErrorWithStdString)
{
    std::string msg = "std::string error";
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_BAD_PARAM, msg);
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), msg.c_str());
}

TEST(PluginLastErrorManagerTest, SetLastErrorSuccessDoesNotChangeError)
{
    const char* prev_msg = "previous error";
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, prev_msg);
    PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_SUCCESS, "should not overwrite");
    EXPECT_STREQ(PluginLastErrorManager::getLastError(), prev_msg);
}
