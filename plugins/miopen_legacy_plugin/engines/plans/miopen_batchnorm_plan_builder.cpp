/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_flatbuffer_type_helpers.hpp>
#include <miopen/miopen.h>
#include <string>

#include "engines/plans/miopen_batchnorm_bwd_plan.hpp"
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
           hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes,
           hipdnn_sdk::data_objects::NodeAttributes_BatchnormBackwardAttributes}))
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

namespace
{

std::string get_node_name(const hipdnn_sdk::data_objects::Node& node)
{
    return node.name() != nullptr ? node.name()->str() : "";
}

void build_plan_inference_single_node(const hipdnnEnginePluginHandle& handle,
                                      const hipdnn_plugin::Graph_interface& op_graph,
                                      const hipdnn_sdk::data_objects::Node& node,
                                      hipdnnEnginePluginExecutionContext& execution_context)
{
    std::ignore = handle;

    const auto* attr = node.attributes_as_BatchnormInferenceAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormInferenceAttributes for node: "
                + get_node_name(node));
    }

    auto params
        = std::make_unique<Batchnorm_fwd_inference_params>(*attr, op_graph.get_tensor_map());
    auto plan = std::make_unique<Batchnorm_fwd_inference_plan>(std::move(params));
    execution_context.set_plan(std::move(plan));
}

void build_plan_bwd_single_node(const hipdnnEnginePluginHandle& handle,
                                const hipdnn_plugin::Graph_interface& op_graph,
                                const hipdnn_sdk::data_objects::Node& node,
                                hipdnnEnginePluginExecutionContext& execution_context)
{
    std::ignore = handle;

    const auto* attr = node.attributes_as_BatchnormBackwardAttributes();
    if(attr == nullptr)
    {
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Failed to convert node attributes to BatchnormBackwardAttributes for node: "
                + get_node_name(node));
    }

    auto params = std::make_unique<Batchnorm_bwd_params>(*attr, op_graph.get_tensor_map());
    auto plan = std::make_unique<Batchnorm_bwd_plan>(std::move(params));
    execution_context.set_plan(std::move(plan));
}

} // namespace

void Miopen_batchnorm_plan_builder::build_plan(
    const hipdnnEnginePluginHandle& handle,
    const hipdnn_plugin::Graph_interface& op_graph,
    hipdnnEnginePluginExecutionContext& execution_context) const
{
    const auto& node = op_graph.get_node(0);

    std::string node_name = get_node_name(node);
    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes_BatchnormInferenceAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd inference plan for node: {}", node_name);
        build_plan_inference_single_node(handle, op_graph, node, execution_context);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes_BatchnormBackwardAttributes:
        HIPDNN_LOG_INFO("Building batchnorm backward plan for node: {}", node_name);
        build_plan_bwd_single_node(handle, op_graph, node, execution_context);
        break;
    default:
        throw hipdnn_plugin::Hipdnn_plugin_exception(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(hipdnn_sdk::data_objects::to_string(node.attributes_type())));
    }
}

} // namespace miopen_legacy_plugin
