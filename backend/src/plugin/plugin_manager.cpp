// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "logging/logging.hpp"

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/engine_heuristic_descriptor.hpp"
#include "fake_plugin.hpp"
#include "hipdnn_exception.hpp"
#include "plugin_manager.hpp"

namespace hipdnn_backend
{

void Plugin_manager::initialize()
{
    // TODO : actually find and init the plugins properly from the plugin directory.
    // For now we will just use a fake plugin.
    _plugins.push_back(std::make_shared<Fake_plugin>());

    for(const auto& plugin : _plugins)
    {
        plugin->set_logging_callback(logging::hipdnn_logging_callback);

        for(const auto& engine_id : plugin->get_engines())
        {
            if(_engine_id_plugin_lookup.contains(engine_id))
            {
                HIPDNN_LOG_ERROR("Plugin_manager::initialize: Duplicate engine_id found: "
                                 + std::to_string(engine_id) + ". Skipping this engine.");
                continue;
            }
            _engine_id_plugin_lookup.insert({engine_id, plugin});
        }
    }
}

std::shared_ptr<Hipdnn_plugin_base> Plugin_manager::get_plugin(int64_t engine_id)
{
    auto plugin_iter = _engine_id_plugin_lookup.find(engine_id);
    if(plugin_iter == _engine_id_plugin_lookup.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
                               "Plugin_manager::get_plugin failed: invalid engineId "
                                   + std::to_string(engine_id));
    }
    return plugin_iter->second;
}

void Plugin_manager::finalize_engine_heuristic(hipdnnBackendDescriptor_t desc)
{
    auto heuristic = desc->as_descriptor<Engine_heuristic_descriptor>();

    auto graph_desc = heuristic->get_graph();

    auto applicable_engines = get_applicable_engines(graph_desc.get());
    std::vector<int64_t> engine_ids(applicable_engines.begin(), applicable_engines.end());

    heuristic->set_engine_ids(engine_ids);
}

void Plugin_manager::finalize_engine_config(hipdnnBackendDescriptor_t desc)
{
    auto config_desc = desc->as_descriptor<Engine_config_descriptor>();

    auto engine_desc = config_desc->get_engine();

    auto graph_desc = engine_desc->get_graph();

    int64_t engine_id = engine_desc->get_engine_id();

    auto plugin = get_plugin(engine_id);

    // TODO - We need to construct + store the engine config fbs with the properties that the user of the API set for the engine config,
    // and then pass that to the plugin as well here.
    auto applicable_engines = plugin->get_applicable_engines(graph_desc.get());
    THROW_IF_FALSE(applicable_engines.contains(engine_id),
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string("Plugin does not support engine id: ") + std::to_string(engine_id));

    config_desc->set_max_workspace_size(
        plugin->get_max_workspace_size(graph_desc.get(), engine_id));
}

std::set<int64_t> Plugin_manager::get_applicable_engines(const Graph_descriptor* graph)
{
    std::set<int64_t> applicable_engines;
    for(const auto& plugin : _plugins)
    {
        auto current_applicable_engines = plugin->get_applicable_engines(graph);

        applicable_engines.insert(current_applicable_engines.begin(),
                                  current_applicable_engines.end());
    }

    return applicable_engines;
}

void Plugin_manager::execute(hipdnnHandle* handle,
                             hipdnnBackendDescriptor_t execution_plan,
                             hipdnnBackendDescriptor_t variant_pack)
{
    auto execution_plan_desc = execution_plan->as_descriptor<Execution_plan_descriptor>();
    auto variant_desc = variant_pack->as_descriptor<Variant_descriptor>();

    THROW_IF_FALSE(execution_plan_desc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Plugin_manager::execute failed: execution_plan_desc is not finalized");

    auto engine_config_desc = execution_plan_desc->get_engine_config();

    auto engine_desc = engine_config_desc->get_engine();

    int64_t engine_id = engine_desc->get_engine_id();

    auto plugin = get_plugin(engine_id);
    assert(plugin != nullptr);

    auto graph_desc = engine_desc->get_graph();

    plugin->execute(graph_desc.get(), variant_desc.get(), handle);
}

} // hipdnn_backend
