// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>

#include <hipdnn_sdk/plugin/flatbuffer_utilities/graph_wrapper.hpp>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>

#include "engine_manager.hpp"
#include "engines/miopen_engine.hpp"

using namespace hipdnn_plugin;

namespace miopen_legacy_plugin
{

Engine_manager::Engine_manager() {}

void Engine_manager::add_engine(std::unique_ptr<Engine_interface> engine)
{
    _engines.emplace(engine->id(), std::move(engine));
}

std::vector<int64_t>
    Engine_manager::get_applicable_engine_ids(const hipdnn_plugin::Graph_interface& op_graph)
{
    std::vector<int64_t> applicable;
    for(const auto& engine : _engines)
    {
        if(engine.second->is_applicable(op_graph))
        {
            applicable.push_back(engine.second->id());
        }
    }
    return applicable;
}

void Engine_manager::get_engine_details(hipdnnEnginePluginHandle& handle,
                                        const hipdnn_plugin::Graph_interface& op_graph,
                                        int64_t engine_id,
                                        hipdnnPluginConstData_t& engine_details_out)
{
    (void)op_graph; // Unused parameter
    auto& engine = get_engine(engine_id);
    engine.get_details(handle, engine_details_out);
}

size_t Engine_manager::get_workspace_size(const hipdnnEnginePluginHandle& handle,
                                          int64_t engine_id,
                                          const hipdnn_plugin::Graph_interface& op_graph) const
{
    auto& engine = get_engine(engine_id);
    return engine.get_workspace_size(handle, op_graph);
}

void Engine_manager::initialize_execution_context(
    const hipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::Graph_interface& op_graph,
    const hipdnn_plugin::Engine_config_interface& engine_config,
    hipdnnEnginePluginExecutionContext& execution_context) const
{
    auto& engine = get_engine(engine_config.engine_id());
    engine.initialize_execution_context(handle, op_graph, execution_context);
}

Engine_interface& Engine_manager::get_engine(int64_t engine_id) const
{
    auto it = _engines.find(engine_id);
    if(it == _engines.end())
    {
        throw Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                      "Engine with ID " + std::to_string(engine_id)
                                          + " not found.");
    }
    return *it->second;
}

}
