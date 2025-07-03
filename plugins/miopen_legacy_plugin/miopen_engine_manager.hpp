// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "engines/miopen_engine.hpp"

class Miopen_engine_manager
{
public:
    Miopen_engine_manager() = default;
    ~Miopen_engine_manager() = default;

    //disallow copy and assignment
    Miopen_engine_manager(const Miopen_engine_manager&) = delete;
    Miopen_engine_manager& operator=(const Miopen_engine_manager&) = delete;

    void initialize_engines();

    std::set<int> get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph);

private:
    std::set<std::shared_ptr<Miopen_engine>> _engines;
};