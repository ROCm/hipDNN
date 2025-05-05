// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plugin_manager.hpp"
#include "fake_plugin.hpp"

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

std::set<int64_t> Plugin_manager::get_applicable_engines(Graph_descriptor* graph,
                                                         Handle* handle /*, Heuristic_Details*/)
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