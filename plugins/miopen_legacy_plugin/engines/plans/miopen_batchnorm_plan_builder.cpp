/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_flatbuffer_type_helpers.hpp>
#include <miopen/miopen.h>

#include "engines/plans/miopen_batchnorm_fwd_inference_plan.hpp"
#include "miopen_batchnorm_plan_builder.hpp"

namespace miopen_legacy_plugin
{

bool Miopen_batchnorm_plan_builder::is_applicable(
    const hipdnn_plugin::Graph_interface& op_graph) const
{

    if(op_graph.node_count() != 1)
    {
        HIPDNN_LOG_INFO(
            "Batchnorm plan builder is applicable only for single node graphs. Graph has {} nodes",
            op_graph.node_count());
        return false;
    }

    if(!op_graph.has_only_supported_attributes(std::set<hipdnn_sdk::data_objects::NodeAttributes>{
           hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes}))
    {
        HIPDNN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
        return false;
    }

    return true;
}

size_t Miopen_batchnorm_plan_builder::get_workspace_size(
    const hipdnnEnginePluginHandle& handle, const hipdnn_plugin::Graph_interface& op_graph) const
{
    std::ignore = handle;
    std::ignore = op_graph;
    //batchnorm plan builder does not require workspace size
    return 0u;
}

void Miopen_batchnorm_plan_builder::build_plan(
    const hipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::Graph_interface& op_graph,
    hipdnnEnginePluginExecutionContext& execution_context) const
{
    std::ignore = handle;

    const auto& node = op_graph.get_node(0);

    if(node.attributes_type()
       == hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes)
    {
        HIPDNN_LOG_INFO("Building batchnorm fwd inference plan for node: {}", node.name()->str());
        if(const auto* batchnorm_inference_attr = node.attributes_as_BatchnormInferenceAttributes();
           batchnorm_inference_attr != nullptr)
        {
            auto fwd_inference_params = std::make_unique<Batchnorm_fwd_inference_params>(
                *batchnorm_inference_attr, op_graph.get_tensor_map());

            auto batchnorm_fwd_plan
                = std::make_unique<Batchnorm_fwd_inference_plan>(std::move(fwd_inference_params));
            execution_context.set_plan(std::move(batchnorm_fwd_plan));
        }
        else
        {
            throw hipdnn_plugin::Hipdnn_plugin_exception(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Failed to convert node attributes to BatchnormInferenceAttributes for node: "
                    + node.name()->str());
        }
    }
    else
    {
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(hipdnn_sdk::data_objects::to_string(node.attributes_type())));
    }
}
}
