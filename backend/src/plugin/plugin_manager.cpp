// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plugin_manager.hpp"
#include "fake_plugin.hpp"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

void Plugin_manager::initialize( // NOLINT(readability-convert-member-functions-to-static)
    /* some stuff to help you find which plugins should be loaded, but for now blank */)
{
    // TODO : actually find and init the plugins properly
    // for now we will just use a fake plugin.
    // # DISCUSS: how do we find the plugins? store in json?
    auto plugin = std::make_shared<Fake_plugin>();

    // for all the applicable engines, add the same plugin to the map
    for(const int64_t& engine_id : plugin->get_applicable_engines(nullptr))
    {
        _plugins.emplace(engine_id, plugin);
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

void Plugin_manager::finalize_engine_config(Engine_config_descriptor* config)
{
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
    auto graph_desc = static_cast<Graph_descriptor*>(graph);
    if(graph_desc == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "hipdnnBackendDescriptor_t is not a valid graph descriptor");
    }

    int64_t engine_id;
    hipdnnBackendGetAttribute(
        engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    // Use the get_plugin helper method instead of iterating through all plugins
    auto plugin = get_plugin(engine_id);
    if(plugin == nullptr)
    {
        throw Hipdnn_exception(
            HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
            std::string("Plugin_manager::finalize_engine_config has invalid engine id: ")
                + std::to_string(engine_id) + " for the given engine config.");
    }

    // Verify that the plugin supports this engine ID
    auto applicable_engines = plugin->get_applicable_engines(graph_desc);
    if(applicable_engines.find(engine_id) == applicable_engines.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               std::string("Plugin does not support engine id: ")
                                   + std::to_string(engine_id));
    }

    // Set the workspace size
    config->set_max_workspace_size(plugin->get_max_workspace_size(graph_desc, engine_id));
}

std::set<int64_t> Plugin_manager::
    get_applicable_engines( // NOLINT(readability-convert-member-functions-to-static)
        Graph_descriptor* graph,
        hipdnnHandle* handle /*, Heuristic_Details*/)
{
    (void)handle;
    std::set<int64_t> applicable_engines;
    for(const auto& [engine_id, plugin] : _plugins)
    {
        auto current_applicable_engines = plugin->get_applicable_engines(graph);

        // Check that there isn't a conflict since Ids must be unique for engines
        applicable_engines.insert(current_applicable_engines.begin(),
                                  current_applicable_engines.end());
    }

    return applicable_engines;
}

void Plugin_manager::execute( // NOLINT(readability-convert-member-functions-to-static)
    Execution_plan_descriptor* execution_plan_desc,
    hipdnnHandle* handle,
    Variant_descriptor* variant_desc)
{
    if(execution_plan_desc == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Plugin_manager::execute failed: execution_plan_desc is null");
    }

    if(handle == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Plugin_manager::execute failed: handle is null");
    }

    if(variant_desc == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Plugin_manager::execute failed: variant_desc is null");
    }

    if(!execution_plan_desc->is_finalized())
    {
        throw Hipdnn_exception(
            HIPDNN_STATUS_BAD_PARAM,
            "Plugin_manager::execute failed: execution_plan_desc is not finalized");
    }
    hipdnnStatus_t status = HIPDNN_STATUS_INTERNAL_ERROR;

    // First get engine config from engine plan descritpor
    hipdnnBackendDescriptor_t engine_config = nullptr;
    status = hipdnnBackendGetAttribute(execution_plan_desc,
                                       HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       &engine_config);

    if(status != HIPDNN_STATUS_SUCCESS)
    {
        throw Hipdnn_exception(
            status, "Plugin_manager::execute failed: could not get engine configuration");
    }

    if(engine_config == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Plugin_manager::execute failed: engine_config is null");
    }

    // second get the engine from the engine config
    hipdnnBackendDescriptor_t engine;
    status = hipdnnBackendGetAttribute(engine_config,
                                       HIPDNN_ATTR_ENGINECFG_ENGINE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       &engine);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        throw Hipdnn_exception(status, "Plugin_manager::execute failed: could not get engine");
    }
    // third get the engine id from the engine
    int64_t engine_id;
    status = hipdnnBackendGetAttribute(
        engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    // Use get_plugin helper to find the plugin for this engine ID
    auto plugin = get_plugin(engine_id);
    if(plugin == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
                               std::string("Plugin_manager::execute has invalid engine id: ")
                                   + std::to_string(engine_id));
    }

    // Then, get the graph from the engine
    hipdnnBackendDescriptor_t graph;
    hipdnnBackendGetAttribute(engine,
                              HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                              HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                              1,
                              nullptr,
                              &graph);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);
    if(graph_desc == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Plugin_manager::execute failed: could not get engine graph");
    }

    // Execute using the plugin
    plugin->execute(graph_desc, variant_desc, handle);
}

}