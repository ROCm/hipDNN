// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Engine
{
public:
    virtual ~Engine() = default;

    virtual int64_t id() const = 0;

    virtual bool is_applicable(const hipdnnPluginConstData_t* op_graph) const = 0;

    virtual size_t get_workspace_size() const = 0;
};

}