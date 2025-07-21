// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/plan_interface.hpp"
#include <gmock/gmock.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Mock_plan : public Plan_interface
{
public:
    MOCK_METHOD(void,
                execute,
                (const hipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                 uint32_t num_device_buffers,
                 void* workspace),
                (const, override));
};

}
