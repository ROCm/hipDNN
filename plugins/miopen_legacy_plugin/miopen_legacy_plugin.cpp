// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <miopen/miopen.h>

#include "hipdnn_sdk/logging/logger.hpp"
#include "hipdnn_sdk/plugin/plugin_api.h"

static const char* _plugin_name = "miopen_legacy_plugin";
static const char* _plugin_version = "1.0.0";

// Implementation of Plugin API

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(!name)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    *name = _plugin_name;

    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(!version)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    *version = _plugin_version;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(!type)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    *type = HIPDNN_PLUGIN_TYPE_ENGINE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(!error_str)
        return;
    *error_str = "No error";
}

} // extern "C"
