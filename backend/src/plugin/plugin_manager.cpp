// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/logging/logger.hpp>

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
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
        for(const auto& engine_id : plugin->get_engines())
        {
            if(_engine_id_plugin_lookup.find(engine_id) != _engine_id_plugin_lookup.end())
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

void Plugin_manager::finalize_engine_config(hipdnnBackendDescriptor_t desc)
{
    assert(desc != nullptr);
    assert(desc->type == HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto config = static_cast<Engine_config_descriptor*>(desc);

    hipdnnBackendDescriptor_t engine;
    hipdnnBackendGetAttribute(
        config, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);

    hipdnnBackendDescriptor_t graph;
    hipdnnBackendGetAttribute(engine,
                              HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                              HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                              1,
                              nullptr,
                              &graph);
    assert(graph != nullptr);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    int64_t engine_id;
    hipdnnBackendGetAttribute(
        engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    auto plugin = get_plugin(engine_id);

    // TODO - We need to construct + store the engine config fbs with the properties that the user of the API set for the engine config,
    // and then pass that to the plugin as well here.
    auto applicable_engines = plugin->get_applicable_engines(graph_desc);
    THROW_IF_EQ(applicable_engines.find(engine_id),
                applicable_engines.end(),
                HIPDNN_STATUS_BAD_PARAM,
                std::string("Plugin does not support engine id: ") + std::to_string(engine_id));

    config->set_max_workspace_size(plugin->get_max_workspace_size(graph_desc, engine_id));
}

std::set<int64_t>
    Plugin_manager::get_applicable_engines(Graph_descriptor* graph,
                                           hipdnnHandle* handle /*, Heuristic_Details*/)
{
    (void)handle;
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
    THROW_IF_NE(execution_plan->type,
                HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Plugin_manager::execute failed: Invalid execution plan descriptor type");

    THROW_IF_NE(variant_pack->type,
                HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Plugin_manager::execute failed: Invalid variant pack descriptor type");

    auto plan_desc = static_cast<Execution_plan_descriptor*>(execution_plan);
    auto variant_desc = static_cast<Variant_descriptor*>(variant_pack);

    THROW_IF_FALSE(plan_desc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Plugin_manager::execute failed: execution_plan_desc is not finalized");

    hipdnnBackendDescriptor_t engine_config = nullptr;

    plan_desc->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                             1,
                             nullptr,
                             &engine_config);
    assert(engine_config != nullptr);

    auto engine_config_desc = static_cast<Engine_config_descriptor*>(engine_config);

    // Get engine directly from engine config
    hipdnnBackendDescriptor_t engine = nullptr;
    engine_config_desc->get_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);
    assert(engine != nullptr);

    auto engine_desc = static_cast<Engine_descriptor*>(engine);

    int64_t engine_id = 0;
    engine_desc->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    auto plugin = get_plugin(engine_id);
    assert(plugin != nullptr);

    hipdnnBackendDescriptor_t graph = nullptr;
    engine_desc->get_attribute(
        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &graph);
    assert(graph != nullptr);

    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    plugin->execute(graph_desc, variant_desc, handle);
}

} // hipdnn_backend
