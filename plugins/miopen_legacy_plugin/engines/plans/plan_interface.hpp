// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class PlanInterface
{
public:
    virtual ~PlanInterface() = default;

    virtual void execute(const hipdnnEnginePluginHandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace = nullptr) const
        = 0;
};

}
