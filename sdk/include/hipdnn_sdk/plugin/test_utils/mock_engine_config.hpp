// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/engine_config_wrapper.hpp>

namespace hipdnn_plugin
{

class Mock_engine_config : public Engine_config_interface
{
public:
    MOCK_METHOD(const hipdnn_sdk::data_objects::EngineConfig&,
                get_engine_config,
                (),
                (const, override));
    MOCK_METHOD(bool, is_valid, (), (const, override));
    MOCK_METHOD(int64_t, engine_id, (), (const, override));
};

}