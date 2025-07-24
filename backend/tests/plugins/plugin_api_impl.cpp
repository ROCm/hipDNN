// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test plugin.

#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_helpers.hpp>

#include "plugin_api_impl.hpp"

using namespace hipdnn_plugin;

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char Plugin_last_error_manager::last_error[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";
static hipdnnCallback_t logging_callback = nullptr;

#ifdef THROW_IF_NULL
#error "THROW_IF_NULL is already defined"
#endif
#define THROW_IF_NULL(value) \
    PLUGIN_THROW_IF_NULL(value, HIPDNN_PLUGIN_STATUS_BAD_PARAM, #value " is null")

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return hipdnn_plugin::try_catch([&]() {
        THROW_IF_NULL(name);
        *name = PLUGIN_NAME;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return hipdnn_plugin::try_catch([&]() {
        THROW_IF_NULL(version);
        *version = PLUGIN_VERSION;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return hipdnn_plugin::try_catch([&]() {
        THROW_IF_NULL(type);
        *type = PLUGIN_TYPE;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    if(callback == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                         "hipdnnPluginGetType: type is null");
    }
    logging_callback = callback;
    logging_callback(HIPDNN_SEV_INFO, "Logging callback successfully set for test plugin.");
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
