// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "handle/handle.hpp"
#include "hipdnn_plugin_base.hpp"

class Engine_config_descriptor;

namespace hipdnn_backend
{

class Plugin_manager
{
    // Populate the hipDNNPlugin's by finding the sources,
    // and constructing the cAPIPlugin, and then constructing
    // the wrapper hipDNNPlugin objects
    void initialize(
        /* some stuff to help you find which plugins should be loaded, but for now blank */);

    // This queries the workspace details for the given engine, and then sets the
    // workspace in the Engine_config_descriptor.  This should throw if
    // an error occurs that prevents the workspace size from being set on the engine
    // config.
    void finalize_engine_config(Engine_config_descriptor* config);

    // Note: Heuristic details is a future thing, for now we can ignore
    // 		 Heuristic details is used to determine sort order on the returned graphs.

    std::set<int64_t> get_applicable_engines(Graph_descriptor* graph,
                                             Handle* handle /*, Heuristic_Details*/);

    // This will redirect the execute to the plugin that owns the engine selected inside the ExecutionPlan
    // Throws if invalid stuff is provided, and later is wrapped with a status + provides message
    void execute(Execution_plan_descriptor* plan, Handle* handle, Variant_descriptor* pack);

private:
    std::vector<std::shared_ptr<Hipdnn_plugin_base>> _plugins;
};

}