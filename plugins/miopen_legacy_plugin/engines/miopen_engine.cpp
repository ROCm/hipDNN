// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_engine.hpp"
#include "solvers/miopen_batchnorm_solver.hpp"

#include <hipdnn_sdk/data_objects/engine_details_generated.h>
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

void Miopen_engine::get_details(hipdnnPluginConstData_t& details_out) const
{
    flatbuffers::FlatBufferBuilder builder;
    auto engine_details = hipdnn_sdk::data_objects::CreateEngineDetails(builder, _id);
    builder.Finish(engine_details);
    auto serialized_details = builder.Release();

    auto* temp_buffer = new uint8_t[serialized_details.size()];
    std::memcpy(temp_buffer, serialized_details.data(), serialized_details.size());

    details_out.ptr = temp_buffer;
    details_out.size = serialized_details.size();
}

size_t Miopen_engine::get_workspace_size(const hipdnnEnginePluginHandle& handle,
                                         const hipdnnPluginConstData_t* op_graph) const
{
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> graph;
    hipdnn_plugin::flatbuffer_utilities::convert_serialized_plugin_graph_to_graph(
        op_graph->ptr, op_graph->size, graph);

    size_t workspace_size = 0;
    for(const auto& solver : _solvers)
    {
        if(solver->is_applicable(*graph))
        {
            workspace_size = solver->get_workspace_size(handle, *graph);
        }
    }
    return workspace_size;
}

void Miopen_engine::add_solver(std::unique_ptr<Solver_interface> solver)
{
    _solvers.insert(std::move(solver));
}

}