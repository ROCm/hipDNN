// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "../attributes/batchnorm_inference_attributes.hpp"
#include "../attributes/graph_attributes.hpp"
#include "../error.hpp"
#include "node.hpp"

namespace hipdnn_frontend::graph
{
class BatchnormInferenceNode : public NodeCRTP<BatchnormInferenceNode> //NOLINT
{
public:
    Batchnorm_inference_attributes attributes;

    BatchnormInferenceNode(Batchnorm_inference_attributes&& batchnorm_attrs,
                           const Graph_attributes&          graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(batchnorm_attrs))
    {
    }

    error_t infer_properties_node() override
    {
        if(attributes.inputs.empty())
        {
            return {error_code_t::INVALID_VALUE,
                    "BatchnormInferenceNode missing input for setting properties"};
        }

        if(attributes.outputs.empty())
        {
            return {error_code_t::INVALID_VALUE,
                    "BatchnormInferenceNode missing output for setting properties"};
        }

        CHECK_HIPDNN_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

        auto x = attributes.inputs[Batchnorm_inference_attributes::input_names::x];
        auto y = attributes.outputs[Batchnorm_inference_attributes::output_names::y];

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        return {};
    }
};
}
