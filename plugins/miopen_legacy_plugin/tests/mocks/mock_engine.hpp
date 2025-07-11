/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#pragma once

#include "engines/engine.hpp"
#include <gmock/gmock.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Mock_engine : public Engine
{
public:
    MOCK_METHOD(int64_t, id, (), (const, override));
    MOCK_METHOD(bool, is_applicable, (const hipdnnPluginConstData_t* op_graph), (const, override));
    MOCK_METHOD(size_t, get_workspace_size, (), (const, override));
};

} // namespace miopen_legacy_plugin