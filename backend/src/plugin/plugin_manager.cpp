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

std::set<int64_t> Plugin_manager::get_applicable_engines( // NOLINT(readability-convert-member-functions-to-static)
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

hipdnnStatus_t Plugin_manager::execute( // NOLINT(readability-convert-member-functions-to-static)
    Execution_plan_descriptor* execution_plan_desc,
    hipdnnHandle* handle,
    Variant_descriptor* variant_desc)
{
    if(execution_plan_desc == nullptr || handle == nullptr || variant_desc == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }

    hipdnnBackendDescriptor_t engine_config_t = nullptr;
    auto plan_desc = dynamic_cast<Execution_plan_descriptor*>(execution_plan_desc);
    plan_desc->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     nullptr,
                                                     &engine_config_t);
    int64_t engine_id = -1;
    Graph_descriptor* graphdesc = nullptr;
    /*
    // engine config stuff
    auto engine_config = dynamic_cast<Engine_config_descriptor*>(engine_config_t);
    if (engine_config == nullptr)
    {
        return HIPDNN_STATUS_BAD_PARAM;
    }
    status = engine_config->get_attribute(
        HIPDNN_ATTR_ENGINE_CONFIG_ENGINE_ID, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return status;
    }

    graphdesc = engine_config_t->get_engine_descriptor()->get_graph();
    */

    // for the given engine id, find the plugin
    auto plugin_iter = _plugins.find(engine_id);
    if(plugin_iter == _plugins.end())
    {
        throw Hipdnn_exception(
            HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
            std::string("Plugin_manager::execute has invalid engine id: ")
                + std::to_string(engine_id) + " for the given engine config.");
    }

    plugin_iter->second->execute(graphdesc, variant_desc, handle);

    return HIPDNN_STATUS_SUCCESS;
}

}