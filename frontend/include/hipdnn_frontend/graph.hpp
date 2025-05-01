// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "flatbuffers/detached_buffer.h"
#include <hipdnn_frontend/attributes/batchnorm_attributes.hpp>
#include <hipdnn_frontend/attributes/batchnorm_inference_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/node/batchnorm_backward_node.hpp>
#include <hipdnn_frontend/node/batchnorm_inference_node.hpp>
#include <hipdnn_frontend/node/batchnorm_node.hpp>
#include <hipdnn_frontend/node/node.hpp>
#include <hipdnn_frontend/node/pointwise_node.hpp>

namespace hipdnn_frontend
{
namespace graph
{
class Graph : public INode
{
private:
    static std::shared_ptr<Tensor_attributes> output_tensor(const std::string& name)
    {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

public:
    // Will be set after building the operation graph.
    // Once we integrate the backend, then we will instead hold onto the descriptor associated with it after building.
    flatbuffers::DetachedBuffer serialized_graph;

    Graph()
        : INode(Graph_attributes{})
    {
    }

    error_t validate()
    {
        return validate_subtree();
    }

    error_t build_operation_graph()
    {
        std::unordered_set<int64_t> used_tensor_uids;
        gather_hipdnn_tensor_ids_subtree(used_tensor_uids);

        std::unordered_map<int64_t, std::shared_ptr<Tensor_attributes>> tensor_lookup;
        int64_t current_tensor_id = 0;

        populate_hipdnn_tensor_ids_subtree(tensor_lookup, current_tensor_id, used_tensor_uids);
        flatbuffers::FlatBufferBuilder builder;

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensor_attributes;
        for(auto& [_, tensor] : tensor_lookup)
        {
            if(tensor)
            {
                tensor_attributes.emplace_back(tensor->pack_attributes(builder));
            }
        }

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        for(auto& node : _sub_nodes)
        {
            if(node)
            {
                nodes.emplace_back(node->pack_node(builder));
            }
        }
        auto graph = hipdnn_sdk::data_objects::CreateGraphDirect(
            builder,
            graph_attributes.get_name().c_str(),
            to_sdk_type(graph_attributes.get_compute_data_type()),
            to_sdk_type(graph_attributes.get_intermediate_data_type()),
            to_sdk_type(graph_attributes.get_io_data_type()),
            &tensor_attributes,
            &nodes);

        builder.Finish(graph);
        serialized_graph = builder.Release();
        // TODO - Lower graph to the backend.
        //        For now we will hold onto the graph, and make it accessible.

        return {};
    }

    const std::string& get_name() const
    {
        return graph_attributes.get_name();
    }
    DataType_t get_compute_data_type() const
    {
        return graph_attributes.get_compute_data_type();
    }
    DataType_t get_intermediate_data_type() const
    {
        return graph_attributes.get_intermediate_data_type();
    }
    DataType_t get_io_data_type() const
    {
        return graph_attributes.get_io_data_type();
    }

    // Forwarding setters
    Graph& set_name(const std::string& name)
    {
        graph_attributes.set_name(name);
        return *this;
    }
    Graph& set_compute_data_type(DataType_t compute_type)
    {
        graph_attributes.set_compute_data_type(compute_type);
        return *this;
    }
    Graph& set_intermediate_data_type(DataType_t intermediate_type)
    {
        graph_attributes.set_intermediate_data_type(intermediate_type);
        return *this;
    }
    Graph& set_io_data_type(DataType_t io_type)
    {
        graph_attributes.set_io_data_type(io_type);
        return *this;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 5>
        batchnorm(const std::shared_ptr<Tensor_attributes>& x,
                  const std::shared_ptr<Tensor_attributes>& scale,
                  const std::shared_ptr<Tensor_attributes>& bias,
                  Batchnorm_attributes attributes)
    {
        auto y = output_tensor(attributes.name + "::Y");
        auto mean_out = output_tensor(attributes.name + "::MEAN");
        auto inv_variance_out = output_tensor(attributes.name + "::INV_VARIANCE");

        auto prev_running_mean = attributes.get_prev_running_mean();
        auto prev_running_variance = attributes.get_prev_running_variance();
        auto momentum = attributes.get_momentum();

        std::shared_ptr<Tensor_attributes> next_running_mean;
        std::shared_ptr<Tensor_attributes> next_running_variance;
        if(prev_running_mean && prev_running_variance && momentum)
        {
            next_running_mean = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
            next_running_variance = output_tensor(attributes.name + "::NEXT_RUNNING_VARIANCE");
        }

        attributes.set_x(x);
        attributes.set_scale(scale);
        attributes.set_bias(bias);
        attributes.set_y(y);
        attributes.set_mean(mean_out);
        attributes.set_inv_variance(inv_variance_out);
        attributes.set_next_running_mean(next_running_mean);
        attributes.set_next_running_variance(next_running_variance);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormNode>(std::move(attributes), graph_attributes));

        return {y, mean_out, inv_variance_out, next_running_mean, next_running_variance};
    }

    std::array<std::shared_ptr<Tensor_attributes>, 3>
        batchnorm_backward(const std::shared_ptr<Tensor_attributes>& dy,
                           const std::shared_ptr<Tensor_attributes>& x,
                           const std::shared_ptr<Tensor_attributes>& scale,
                           Batchnorm_backward_attributes attributes)
    {
        auto dx = output_tensor(attributes.name + "::DX");
        attributes.set_dx(dx);

        auto dscale = output_tensor(attributes.name + "::DSCALE");
        attributes.set_dscale(dscale);

        auto dbias = output_tensor(attributes.name + "::DBIAS");
        attributes.set_dbias(dbias);

        attributes.set_x(x);
        attributes.set_dy(dy);
        attributes.set_scale(scale);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormBackwardNode>(std::move(attributes), graph_attributes));

        return {dx, dscale, dbias};
    }

    std::shared_ptr<Tensor_attributes>
        batchnorm_inference(const std::shared_ptr<Tensor_attributes>& x,
                            const std::shared_ptr<Tensor_attributes>& mean,
                            const std::shared_ptr<Tensor_attributes>& inv_variance,
                            const std::shared_ptr<Tensor_attributes>& scale,
                            const std::shared_ptr<Tensor_attributes>& bias,
                            Batchnorm_inference_attributes attributes)
    {
        auto y = attributes.outputs[Batchnorm_inference_attributes::output_names::y]
            = output_tensor(attributes.name + "::Y");
        attributes.inputs[Batchnorm_inference_attributes::input_names::x] = x;
        attributes.inputs[Batchnorm_inference_attributes::input_names::mean] = mean;
        attributes.inputs[Batchnorm_inference_attributes::input_names::inv_variance] = inv_variance;
        attributes.inputs[Batchnorm_inference_attributes::input_names::scale] = scale;
        attributes.inputs[Batchnorm_inference_attributes::input_names::bias] = bias;

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormInferenceNode>(std::move(attributes), graph_attributes));

        return y;
    }

    std::shared_ptr<Tensor_attributes> pointwise(const std::shared_ptr<Tensor_attributes>& in_0,
                                                 Pointwise_attributes attributes)

    {
        if(attributes.name.empty())
        {
            attributes.name = "Pointwise_" + std::to_string(_sub_nodes.size());
        }
        if(in_0->get_name().empty())
        {
            in_0->set_name(attributes.name + "::IN_0");
        }
        auto out_0 = attributes.outputs[Pointwise_attributes::output_names::out_0]
            = output_tensor(attributes.name + "::OUT_0");
        attributes.inputs[Pointwise_attributes::input_names::in_0] = in_0;

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out_0;
    }

    std::shared_ptr<Tensor_attributes> pointwise(const std::shared_ptr<Tensor_attributes>& in_0,
                                                 const std::shared_ptr<Tensor_attributes>& in_1,
                                                 Pointwise_attributes attributes)

    {
        if(attributes.name.empty())
        {
            attributes.name = "Pointwise_" + std::to_string(_sub_nodes.size());
        }
        if(in_0->get_name().empty())
        {
            in_0->set_name(attributes.name + "::IN_0");
        }
        if(in_1->get_name().empty())
        {
            in_1->set_name(attributes.name + "::IN_1");
        }
        auto out_0 = attributes.outputs[Pointwise_attributes::output_names::out_0]
            = output_tensor(attributes.name + "::OUT_0");
        attributes.inputs[Pointwise_attributes::input_names::in_0] = in_0;
        attributes.inputs[Pointwise_attributes::input_names::in_1] = in_1;

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out_0;
    }

    std::shared_ptr<Tensor_attributes> pointwise(const std::shared_ptr<Tensor_attributes>& in_0,
                                                 const std::shared_ptr<Tensor_attributes>& in_1,
                                                 const std::shared_ptr<Tensor_attributes>& in_2,
                                                 Pointwise_attributes attributes)

    {
        if(attributes.name.empty())
        {
            attributes.name = "Pointwise_" + std::to_string(_sub_nodes.size());
        }
        if(in_0->get_name().empty())
        {
            in_0->set_name(attributes.name + "::IN_0");
        }
        if(in_1->get_name().empty())
        {
            in_1->set_name(attributes.name + "::IN_1");
        }
        if(in_2->get_name().empty())
        {
            in_2->set_name(attributes.name + "::IN_2");
        }
        auto out_0 = attributes.outputs[Pointwise_attributes::output_names::out_0]
            = output_tensor(attributes.name + "::OUT_0");
        attributes.inputs[Pointwise_attributes::input_names::in_0] = in_0;
        attributes.inputs[Pointwise_attributes::input_names::in_1] = in_1;
        attributes.inputs[Pointwise_attributes::input_names::in_2] = in_2;

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out_0;
    }
};
}
}