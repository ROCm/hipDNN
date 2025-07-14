// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/utilities/string_util.hpp>

#include <string>

// NOTE: The last_error variable must be defined in one of the plugin source files:
//
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
// thread_local char Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

namespace hipdnn_plugin
{

class Plugin_last_error_manager
{
private:
    // We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
    // This prevents the shared object (plugin) from being unloaded until the program terminates.
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    thread_local static char last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH];

public:
    static hipdnnPluginStatus_t set_last_error(hipdnnPluginStatus_t status, const char* message)
    {
        if(status == HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            return status;
        }

        // TODO spdlog prevents the library from being unloaded (PluginManagerTest.LastErrorOnSecondLoad is failing).
        // This line can be uncommented once the root cause is identified and fixed.
        //HIPDNN_LOG_ERROR("Error occured in status:{} message:{}", status, message);

        hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
            last_error, message, sizeof(last_error));

        return status;
    }

    static hipdnnPluginStatus_t set_last_error(hipdnnPluginStatus_t status,
                                               const std::string& message)
    {
        return set_last_error(status, message.c_str());
    }

    static const char* get_last_error()
    {
        return last_error;
    }
};

}