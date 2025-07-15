// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>
#include <unordered_map>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "engines/engine.hpp"

namespace miopen_legacy_plugin
{

class Engine_manager
{
public:
    Engine_manager();
    ~Engine_manager() = default;

    //disallow copy and assignment
    Engine_manager(const Engine_manager&) = delete;
    Engine_manager& operator=(const Engine_manager&) = delete;

    void add_engine(std::unique_ptr<Engine> engine);

    std::set<int64_t> get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph);

private:
    std::set<std::unique_ptr<Engine>> _engines;
};

}