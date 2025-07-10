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
    auto graph_ptr = hipdnn_sdk::data_objects::GetGraph(op_graph->ptr);

    for(const auto& solver : _solvers)
    {
        if(solver->is_applicable(*graph_ptr))
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
    auto graph_ptr = hipdnn_sdk::data_objects::GetGraph(op_graph->ptr);

    size_t workspace_size = 0;
    for(const auto& solver : _solvers)
    {
        if(solver->is_applicable(*graph_ptr))
        {
            workspace_size = solver->get_workspace_size(handle, *graph_ptr);
        }
    }
    return workspace_size;
}

void Miopen_engine::execute_graph(const hipdnnEnginePluginHandle& handle,
                                  const hipdnnEnginePluginExecutionContext& execution_context,
                                  const hipdnnPluginDeviceBuffer_t* device_buffers,
                                  uint32_t num_device_buffers,
                                  void* workspace) const
{

    for(const auto& solver : _solvers)
    {

        //todo, add override for this kind of is applicable check
        // if(solver->is_applicable(*graph_ptr))
        // {
        solver->execute_graph(
            handle, execution_context, device_buffers, num_device_buffers, workspace);
        //}
    }
}

void Miopen_engine::add_solver(std::unique_ptr<Solver_interface> solver)
{
    _solvers.insert(std::move(solver));
}

}