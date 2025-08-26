// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>
#include <hipdnn_sdk/utilities/string_util.hpp>

// NOTE: The last_error variable must be defined in one of the plugin source files:
//
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
// thread_local char PluginLastErrorManager::_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

namespace hipdnn_plugin
{

class PluginLastErrorManager
{
private:
    // We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
    // This prevents the shared object (plugin) from being unloaded until the program terminates.
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    thread_local static char _lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH];

public:
    static hipdnnPluginStatus_t setLastError(hipdnnPluginStatus_t status, const char* message)
    {
        if(status == HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            return status;
        }

        HIPDNN_LOG_ERROR("Error occured in status:{} message:{}", status, message);

        hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
            _lastError, message, sizeof(_lastError));

        return status;
    }

    static hipdnnPluginStatus_t setLastError(hipdnnPluginStatus_t status,
                                             const std::string& message)
    {
        return setLastError(status, message.c_str());
    }

    static const char* getLastError()
    {
        return _lastError;
    }
};

}
