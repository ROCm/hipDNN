// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_sdk/logging/logger.hpp"
#include "hipdnn_sdk/plugin/plugin_api_data_types.h"
#include "hipdnn_sdk/utilities/string_util.hpp"

#include <string>

namespace hipdnn_plugin
{
// TODO replace with using the define from a common SDK header.
static const size_t HIPDNN_MAX_ERROR_STRING_SIZE = 256;

inline const char* hipdnn_plugin_get_status_string(hipdnnPluginStatus_t status)
{
    switch(status)
    {
    case HIPDNN_PLUGIN_STATUS_SUCCESS:
        return "HIPDNN_PLUGIN_STATUS_SUCCESS";
    case HIPDNN_PLUGIN_STATUS_BAD_PARAM:
        return "HIPDNN_PLUGIN_STATUS_BAD_PARAM";
    case HIPDNN_PLUGIN_INVALID_VALUE:
        return "HIPDNN_PLUGIN_INVALID_VALUE";
    case HIPDNN_PLUGIN_INTERNAL_ERROR:
        return "HIPDNN_PLUGIN_INTERNAL_ERROR";
    default:
        return "HIPDNN_PLUGIN_STATUS_UNKNOWN";
    }
}

// TODO
// Discuss moving this class elsewhere as the thread local storage must be instantiated in a
// cpp file.  Its a duplicate symbol error if done all in the same header file.
// For now all that need to occur is anywhere that uses this file must instantiate the
// thread local storage in a cpp file, a small price to pay for the convenience of
// making this code shared.
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

        HIPDNN_LOG_ERROR("Error occured in status:{} message:{}",
                         hipdnn_plugin_get_status_string(status),
                         message);

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