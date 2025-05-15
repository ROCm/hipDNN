// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <miopen/miopen.h>

#include "hipdnn_sdk/plugin/plugin_api.h"

static const char* _plugin_name = "miopen_legacy_plugin";
static const char* _plugin_version = "1.0.0";

// Implementation of Plugin API

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(!name)
        return hipdnnPluginStatusBadParam;
    *name = _plugin_name;
    return hipdnnPluginStatusSuccess;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(!version)
        return hipdnnPluginStatusBadParam;
    *version = _plugin_version;
    return hipdnnPluginStatusSuccess;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(!type)
        return hipdnnPluginStatusBadParam;
    *type = hipdnnPluginTypeEngine;
    return hipdnnPluginStatusSuccess;
}

hipdnnPluginStatus_t hipdnnPluginGetNumEngines(unsigned* num_engines)
{
    if(!num_engines)
        return hipdnnPluginStatusBadParam;
    *num_engines = 1;
    return hipdnnPluginStatusSuccess;
}

hipdnnPluginStatus_t hipdnnPluginRunEngine(unsigned engine_index,
                                           const uint32_t* input,
                                           uint32_t* output,
                                           uint32_t size)
{
    if(engine_index != 0 || !input || !output)
        return hipdnnPluginStatusBadParam;
    // Dummy implementation: just copy input to output
    for(uint32_t i = 0; i < size; ++i)
    {
        output[i] = input[i];
    }
    return hipdnnPluginStatusSuccess;
}

} // extern "C"
