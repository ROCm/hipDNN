// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple plugin.
// It is used to test the plugin system and ensure that plugins can be loaded and
// unloaded correctly.

#include <string>

#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/utilities/string_util.hpp>

static const char* const PLUGIN_NAME = "Plugin2";
static const char* const PLUGIN_VERSION = "2.0";
static const hipdnnPluginType_t PLUGIN_TYPE = HIPDNN_PLUGIN_TYPE_UNSPECIFIED;

// We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
// This prevents the shared object (plugin) from being unloaded until the program terminates.
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local static char last_error_string[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

static void set_last_error_string(const std::string& error)
{
    hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
        last_error_string, error.c_str(), sizeof(last_error_string));
}

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        set_last_error_string("hipdnnPluginGetName: name is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        set_last_error_string("hipdnnPluginGetVersion: version is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        set_last_error_string("hipdnnPluginGetType: type is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
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
    *error_str = last_error_string;
}
