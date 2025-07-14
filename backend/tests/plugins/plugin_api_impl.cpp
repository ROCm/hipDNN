// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test plugin.

#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include "plugin_api_impl.hpp"

using namespace hipdnn_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                         "hipdnnPluginGetName: name is null");
    }
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                         "hipdnnPluginGetVersion: version is null");
    }
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                         "hipdnnPluginGetType: type is null");
    }
    *type = PLUGIN_TYPE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str == nullptr)
    {
        return;
    }
    *error_str = Plugin_last_error_manager::get_last_error();
}
