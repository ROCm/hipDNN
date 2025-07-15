// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/utilities/string_util.hpp>

#include <string>

namespace hipdnn_plugin
{
// TODO replace with using the define from a common SDK header.
static const size_t HIPDNN_MAX_ERROR_STRING_SIZE = 256;

// NOTE, PLEASE READ
// This class defines a thread local storage for the last error message.  This thread storage
// must be instantiated in a cpp file or you will get duplicate symbol errors. All you need to do
// to use this class is add the following lines to any cpp file in your plugin.
//
// // NOLINTNEXTLINE(modernize-avoid-c-arrays)
// thread_local char Plugin_last_error_manager::last_error[HIPDNN_MAX_ERROR_STRING_SIZE] = "";
//
// This is sligtly inconvenient, but its a small price to pay in order to keep this reusable.
class Plugin_last_error_manager
{
private:
    // We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
    // This prevents the shared object (plugin) from being unloaded until the program terminates.
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    thread_local static char last_error[HIPDNN_MAX_ERROR_STRING_SIZE];

public:
    static hipdnnPluginStatus_t set_last_error(hipdnnPluginStatus_t status, const char* message)
    {
        if(status == HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            return status;
        }

        HIPDNN_LOG_ERROR("Error occured in status:{} message:{}", status, message);

        hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
            last_error, message, HIPDNN_MAX_ERROR_STRING_SIZE);

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