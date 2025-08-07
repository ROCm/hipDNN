// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Plan_interface
{
public:
    virtual ~Plan_interface() = default;

    virtual void execute(const hipdnnEnginePluginHandle& handle,
                         const hipdnnPluginDeviceBuffer_t* device_buffers,
                         uint32_t num_device_buffers,
                         void* workspace = nullptr) const
        = 0;
};

}
