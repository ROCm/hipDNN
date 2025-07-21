// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_engine.hpp"
#include "plans/miopen_batchnorm_plan_builder.hpp"

#include <hipdnn_sdk/data_objects/engine_details_generated.h>

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

bool Miopen_engine::is_applicable(const hipdnn_plugin::Graph_interface& op_graph) const
{
    // This is wrong if we ever have more than 1 plan builder thats applicable.
    // If this is the case, we should split plan builders accross multiple engines.
    for(const auto& plan_builder : _plan_builders)
    {
        if(plan_builder->is_applicable(op_graph))
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
                                         const hipdnn_plugin::Graph_interface& op_graph) const
{
    size_t workspace_size = 0;
    for(const auto& plan_builder : _plan_builders)
    {
        if(plan_builder->is_applicable(op_graph))
        {
            workspace_size
                = std::max(workspace_size, plan_builder->get_workspace_size(handle, op_graph));
        }
    }
    return workspace_size;
}

void Miopen_engine::initialize_execution_context(
    const hipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::Graph_interface& op_graph,
    hipdnnEnginePluginExecutionContext& execution_context) const
{
    for(const auto& plan_builder : _plan_builders)
    {
        if(plan_builder->is_applicable(op_graph))
        {
            plan_builder->build_plan(handle, op_graph, execution_context);
            break;
        }
    }
}

void Miopen_engine::add_plan_builder(std::unique_ptr<Plan_builder_interface> plan_builder)
{
    _plan_builders.insert(std::move(plan_builder));
}

}