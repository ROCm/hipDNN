// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "miopen_engine.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

class Miopen_fake_engine : public Miopen_engine
{
public:
    Miopen_fake_engine(int64_t id);

    int64_t id() const override;

    bool is_applicable(const hipdnnPluginConstData_t*) const override;
    size_t get_workspace_size() const override;

private:
    int64_t _id;
};