// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engine_manager.hpp"
#include "engines/miopen_engine.hpp"

#include <algorithm>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

using namespace hipdnn_plugin;

namespace miopen_legacy_plugin
{

Engine_manager::Engine_manager() {}

void Engine_manager::add_engine(std::unique_ptr<Engine_interface> engine)
{
    _engines.emplace(engine->id(), std::move(engine));
}

std::set<int64_t> Engine_manager::get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph)
{
    std::set<int64_t> applicable;
    for(const auto& engine : _engines)
    {
        if(engine.second->is_applicable(op_graph))
        {
            applicable.insert(engine.second->id());
        }
    }
    return applicable;
}

void Engine_manager::get_engine_details(const hipdnnPluginConstData_t* op_graph,
                                        int64_t engine_id,
                                        hipdnnPluginConstData_t& engine_details_out)
{
    (void)op_graph; // Unused parameter

    auto engine = _engines.find(engine_id);
    if(engine == _engines.end())
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_INVALID_VALUE,
                                      "Engine with ID " + std::to_string(engine_id)
                                          + " not found.");
    }
    engine->second->get_details(engine_details_out);
}

size_t Engine_manager::get_workspace_size(const hipdnnEnginePluginHandle& handle,
                                          int64_t engine_id,
                                          const hipdnnPluginConstData_t* op_graph) const
{
    auto it = _engines.find(engine_id);
    if(it == _engines.end())
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_INVALID_VALUE,
                                      "Engine with ID " + std::to_string(engine_id)
                                          + " not found.");
    }
    return it->second->get_workspace_size(handle, op_graph);
}
}