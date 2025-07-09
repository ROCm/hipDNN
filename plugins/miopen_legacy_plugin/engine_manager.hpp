// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

#include "engines/engine_interface.hpp"

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

    void add_engine(std::unique_ptr<Engine_interface> engine);

    std::vector<int64_t> get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph);

    void get_engine_details(const hipdnnPluginConstData_t* op_graph,
                            int64_t engine_id,
                            hipdnnPluginConstData_t& engine_details_out);

    size_t get_workspace_size(const hipdnnEnginePluginHandle& handle,
                              int64_t engine_id,
                              const hipdnnPluginConstData_t* op_graph) const;

private:
    std::unordered_map<int64_t, std::unique_ptr<Engine_interface>> _engines;
};

}