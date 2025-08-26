// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_engine.hpp"
#include "plans/miopen_batchnorm_plan_builder.hpp"

#include <hipdnn_sdk/data_objects/engine_details_generated.h>

namespace miopen_legacy_plugin
{

MiopenEngine::MiopenEngine(int64_t id)
    : _id(id)
{
}

int64_t MiopenEngine::id() const
{
    return _id;
}

bool MiopenEngine::isApplicable(const hipdnn_plugin::Graph_interface& opGraph) const
{
    // This is wrong if we ever have more than 1 plan builder thats applicable.
    // If this is the case, we should split plan builders accross multiple engines.
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(opGraph))
        {
            return true;
        }
    }
    return false;
}

void MiopenEngine::getDetails(hipdnnEnginePluginHandle& handle,
                              hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetails = hipdnn_sdk::data_objects::CreateEngineDetails(builder, _id);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.store_engine_details_detached_buffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t MiopenEngine::getWorkspaceSize(const hipdnnEnginePluginHandle& handle,
                                      const hipdnn_plugin::Graph_interface& opGraph) const
{
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(opGraph))
        {
            workspaceSize
                = std::max(workspaceSize, planBuilder->getWorkspaceSize(handle, opGraph));
        }
    }
    return workspaceSize;
}

void MiopenEngine::initializeExecutionContext(
    const hipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::Graph_interface& opGraph,
    hipdnnEnginePluginExecutionContext& executionContext) const
{
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, executionContext);
            break;
        }
    }
}

void MiopenEngine::add_plan_builder(std::unique_ptr<PlanBuilderInterface> planBuilder)
{
    _planBuilders.insert(std::move(planBuilder));
}

}
