// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "../attributes/graph_attributes.hpp"
#include "../attributes/pointwise_attributes.hpp"
#include "../error.hpp"
#include "../utilities.hpp"
#include "node.hpp"

namespace hipdnn_frontend::graph
{
class PointwiseNode : public NodeCRTP<PointwiseNode> // NOLINT
{
public:
    Pointwise_attributes attributes;

    PointwiseNode(Pointwise_attributes&& batchnorm_attrs, const Graph_attributes& graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(batchnorm_attrs))
    {
    }

    error_t infer_properties_node() override
    {
        if(attributes.inputs.empty())
        {
            return {error_code_t::INVALID_VALUE,
                    "PointwiseNode missing input for setting properties"};
        }

        if(attributes.outputs.empty())
        {
            return {error_code_t::INVALID_VALUE,
                    "PointwiseNode missing output for setting properties"};
        }

        CHECK_HIPDNN_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

        auto x   = attributes.inputs[Pointwise_attributes::input_names::in_0];
        auto out = attributes.outputs[Pointwise_attributes::output_names::out_0];

        if(out->get_dim().empty())
        {
            std::vector<std::vector<int64_t>> input_shapes;
            for(auto& [_, tensor] : attributes.inputs)
            {
                if(tensor)
                {
                    input_shapes.push_back(tensor->get_dim());
                }
            }

            auto output_dims = out->get_dim();
            CHECK_HIPDNN_ERROR(find_common_shape(input_shapes, output_dims));
            out->set_dim(output_dims);
        }

        // Note: Strides will only be set if there is an input tensor with the same dims currently.
        //       This could fail if the input tensors were something like (1, 1, 2) and (2, 1, 1).
        if(out->get_stride().empty())
        {
            for(auto& [_, tensor] : attributes.inputs)
            {
                if(tensor && tensor->get_dim() == out->get_dim())
                {
                    out->set_stride(tensor->get_stride());
                    break;
                }
            }

            if(out->get_stride().empty())
            {
                return {error_code_t::INVALID_VALUE, "PointwiseNode output missing stride"};
            }
        }

        return {};
    }
};
}
