// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "handle/handle.hpp"
#include "hipdnn_plugin_base.hpp"
#include <map>

namespace hipdnn_backend
{

struct Plugin_manager
{
    // Populate the hipDNNPlugin's by finding the sources,
    // and constructing the cAPIPlugin, and then constructing
    // the wrapper hipDNNPlugin objects
    void initialize(/* heuristics */);

    // This queries the workspace details for the given engine config, and then sets the
    // workspace in the Engine_config_descriptor.  This should throw if
    // an error occurs that prevents the workspace size from being set on the engine
    // config.
    void finalize_engine_config(hipdnnBackendDescriptor_t desc);

    // This populates the ordered results of the applicable engines inside the heuristic descriptor.
    void finalize_engine_heuristic(hipdnnBackendDescriptor_t desc);

    // Note: Heuristic details is a future thing, for now we can ignore
    // 		 Heuristic details is used to determine sort order on the returned graphs.

    std::set<int64_t> get_applicable_engines(const Graph_descriptor* graph);

    // This will redirect the execute to the plugin that owns the engine selected inside the ExecutionPlan
    // Throws if invalid stuff is provided, and later is wrapped with a status + provides message
    void execute(hipdnnHandle* handle,
                 hipdnnBackendDescriptor_t execution_plan,
                 hipdnnBackendDescriptor_t variant_pack);

private:
    std::map<int64_t, std::shared_ptr<Hipdnn_plugin_base>> _engine_id_plugin_lookup;
    std::vector<std::shared_ptr<Hipdnn_plugin_base>> _plugins;

    std::shared_ptr<Hipdnn_plugin_base> get_plugin(int64_t engine_id);
};

}