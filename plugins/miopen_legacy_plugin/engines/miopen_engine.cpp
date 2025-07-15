// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_engine.hpp"
#include "solvers/miopen_batchnorm_solver.hpp"

#include <hipdnn_sdk/plugin/plugin_flatbuffer_utilities.hpp>

namespace miopen_legacy_plugin
{

Miopen_engine::Miopen_engine(int64_t id)
    : _id(id)
{
}

int64_t Miopen_engine::id() const
{
    return _id;
}

bool Miopen_engine::is_applicable(const hipdnnPluginConstData_t* op_graph) const
{

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    hipdnn_plugin::flatbuffer_utilities::convert_serialized_plugin_graph_to_graph(
        op_graph->ptr, op_graph->size, graph);

    for(const auto& solver : _solvers)
    {
        if(solver->is_applicable(*graph))
        {
            return true;
        }
    }
    return false;
}

size_t Miopen_engine::get_workspace_size() const
{
    return 1337;
}

void Miopen_engine::add_solver(std::unique_ptr<Solver> solver)
{
    _solvers.insert(std::move(solver));
}

}