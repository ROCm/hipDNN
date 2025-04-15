// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple plugin for the engine.
// It is used to test the plugin system and ensure that plugins can be loaded and
// unloaded correctly.

#include "plugin_api.h"

static const char*              PLUGIN_NAME        = "EnginePlugin2";
static const char*              PLUGIN_VERSION     = "2.0";
static const hipdnnPluginType_t PLUGIN_TYPE        = hipdnnPluginTypeEngine;
static const unsigned           PLUGIN_NUM_ENGINES = 3;

// Each engine plugin must implement the following functions:

extern "C" const char* hipdnnPluginGetName()
{
    return PLUGIN_NAME;
}

extern "C" const char* hipdnnPluginGetVersion()
{
    return PLUGIN_VERSION;
}

extern "C" hipdnnPluginType_t hipdnnPluginGetType()
{
    return PLUGIN_TYPE;
}

extern "C" unsigned hipdnnPluginGetNumEngines()
{
    return PLUGIN_NUM_ENGINES;
}

extern "C" int hipdnnPluginRunEngine(unsigned engine_index, int input)
{
    if(engine_index >= PLUGIN_NUM_ENGINES)
    {
        return -1; // Invalid engine number
    }
    return input * 2;
}
