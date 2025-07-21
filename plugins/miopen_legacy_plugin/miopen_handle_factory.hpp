// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Miopen_handle_factory
{
public:
    static void create_miopen_handle(hipdnnEnginePluginHandle_t* handle);

    static void destroy_miopen_handle(hipdnnEnginePluginHandle_t handle);
};

}
