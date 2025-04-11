// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes/batchnorm_inference_attributes.hpp"
#include "attributes/pointwise_attributes.hpp"
#include "node/batchnorm_inference_node.hpp"
#include "node/node.hpp"
#include "node/pointwise_node.hpp"

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
    Graph()
        : INode(Graph_attributes{})
    {
    }

    const std::string& get_name() const
    {
        return graph_attributes.get_name();
    }
    DataType_t get_compute_type() const
    {
        return graph_attributes.get_compute_data_type();
    }
    DataType_t get_intermediate_type() const
    {
        return graph_attributes.get_intermediate_data_type();
    }
    DataType_t get_io_type() const
    {
        return graph_attributes.get_io_data_type();
    }

    // Forwarding setters
    Graph& set_name(const std::string& name)
    {
        graph_attributes.set_name(name);
        return *this;
    }
    Graph& set_compute_type(DataType_t compute_type)
    {
        graph_attributes.set_compute_data_type(compute_type);
        return *this;
    }
    Graph& set_intermediate_type(DataType_t intermediate_type)
    {
        graph_attributes.set_intermediate_data_type(intermediate_type);
        return *this;
    }
    Graph& set_io_type(DataType_t io_type)
    {
        graph_attributes.set_io_data_type(io_type);
        return *this;
    }

    std::shared_ptr<Tensor_attributes>
        batchnorm_inference(const std::shared_ptr<Tensor_attributes>& x,
                            const std::shared_ptr<Tensor_attributes>& mean,
                            const std::shared_ptr<Tensor_attributes>& inv_variance,
                            const std::shared_ptr<Tensor_attributes>& scale,
                            const std::shared_ptr<Tensor_attributes>& bias,
                            Batchnorm_inference_attributes            attributes)
    {
        auto y = attributes.outputs[Batchnorm_inference_attributes::output_names::y]
            = output_tensor(attributes.name + "::Y");
        attributes.inputs[Batchnorm_inference_attributes::input_names::x]            = x;
        attributes.inputs[Batchnorm_inference_attributes::input_names::mean]         = mean;
        attributes.inputs[Batchnorm_inference_attributes::input_names::inv_variance] = inv_variance;
        attributes.inputs[Batchnorm_inference_attributes::input_names::scale]        = scale;
        attributes.inputs[Batchnorm_inference_attributes::input_names::bias]         = bias;

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