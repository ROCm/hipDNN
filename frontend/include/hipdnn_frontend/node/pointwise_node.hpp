// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "node.hpp"
#include <hipdnn_frontend/attributes/graph_attributes.hpp>
#include <hipdnn_frontend/attributes/pointwise_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

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

    error_t pre_validate_node() const override
    {
        if(!attributes.get_input_0())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing IN_0 for pre-validation"};
        }
        if(!attributes.get_output_0())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing OUT_0 for pre-validation"};
        }
        if(attributes.get_mode() == PointwiseMode_t::NOT_SET)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing operation for pre-validation"};
        }

        return {};
    }

    error_t infer_properties_node() override
    {
        if(attributes.inputs.empty())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing input for setting properties"};
        }

        if(attributes.outputs.empty())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing output for setting properties"};
        }

        CHECK_HIPDNN_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

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
                return {error_code_t::ATTRIBUTE_NOT_SET, "PointwiseNode output missing stride"};
            }
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.name.c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_PointwiseAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
}
