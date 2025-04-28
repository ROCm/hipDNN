// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple plugin.
// It is used to test the plugin system and ensure that plugins can be loaded and
// unloaded correctly.

#include <hipdnn_sdk/plugin/plugin_api.h>

static const char* const        PLUGIN_NAME    = "Plugin2";
static const char* const        PLUGIN_VERSION = "2.0";
static const hipdnnPluginType_t PLUGIN_TYPE    = hipdnnPluginTypeUnspecified;

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *name = PLUGIN_NAME;
    return hipdnnPluginStatusSuccess;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *version = PLUGIN_VERSION;
    return hipdnnPluginStatusSuccess;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *type = PLUGIN_TYPE;
    return hipdnnPluginStatusSuccess;
}
