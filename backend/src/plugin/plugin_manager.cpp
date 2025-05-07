// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plugin_manager.hpp"
#include "fake_plugin.hpp"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{

void Plugin_manager::initialize(
    /* some stuff to help you find which plugins should be loaded, but for now blank */)
{
    // TODO : actually find and init the plugins properly
    // for now we will just use a fake plugin.
    // # DISCUSS: how do we find the plugins?
    _plugins.push_back(std::make_shared<Fake_plugin>());
}

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
    auto graph_desc = dynamic_cast<Graph_descriptor*>(graph);
    if(graph_desc == nullptr)
    {
        throw hipdnn_backend::Hipdnn_exception(
            HIPDNN_STATUS_BAD_PARAM, "hipdnnBackendDescriptor_t is not a valid graph descriptor");
    }

    int64_t engine_id;
    hipdnnBackendGetAttribute(
        engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    for(auto& plugin : _plugins)
    {
        auto applicable_engines = plugin->get_applicable_engines(graph_desc);
        if(applicable_engines.find(engine_id) != applicable_engines.end())
        {
            config->set_max_workspace_size(plugin->get_max_workspace_size(graph_desc, engine_id));
            return;
        }
    }
}

std::set<int64_t>
    Plugin_manager::get_applicable_engines(Graph_descriptor* graph,
                                           hipdnnHandle* handle /*, Heuristic_Details*/)
{
    (void)handle;
    std::set<int64_t> applicable_engines;
    for(auto& plugin : _plugins)
    {
        auto current_applicable_engines = plugin->get_applicable_engines(graph);

        // Check that there isn't a conflict since Ids must be unique for engines
        applicable_engines.insert(current_applicable_engines.begin(),
                                  current_applicable_engines.end());
    }

    return applicable_engines;
}

}