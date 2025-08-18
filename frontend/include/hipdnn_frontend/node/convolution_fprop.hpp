// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "node.hpp"
#include <hipdnn_frontend/attributes/convolution_fwd_attributes.hpp>
#include <hipdnn_frontend/attributes/graph_attributes.hpp>
#include <hipdnn_frontend/error.hpp>
#include <hipdnn_frontend/utilities.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/shape_utils.hpp>

namespace hipdnn_frontend::graph
{
class ConvolutionNode : public NodeCRTP<ConvolutionNode> //NOLINT
{
public:
    Convolution_fprop_attributes attributes;

    ConvolutionNode(Convolution_fprop_attributes&& conv_attrs, const Graph_attributes& graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(conv_attrs))
    {
    }

    error_t pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing x (input) for pre-validation"};
        }
        if(!attributes.get_w())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing w (weights) for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing y (output) for pre-validation"};
        }

        // Validate convolution parameters
        if(attributes.get_pre_padding().empty())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing pre_padding for pre-validation"};
        }
        if(attributes.get_post_padding().empty())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing post_padding for pre-validation"};
        }
        if(attributes.get_stride().empty())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing stride for pre-validation"};
        }
        if(attributes.get_dilation().empty())
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing dilation for pre-validation"};
        }

        return {};
    }

    error_t infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto w = attributes.get_w();
        auto y = attributes.get_y();

        if(!x)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing x for setting properties"};
        }

        if(!w)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing w for setting properties"};
        }

        if(!y)
        {
            return {error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode missing y for setting properties"};
        }

        CHECK_HIPDNN_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

        // Infer output dimensions if not set
        if(y->get_dim().empty())
        {
            // TODO
        }

        // Infer output strides if not set
        if(y->get_stride().empty())
        {
            // TODO
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.name.c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_ConvolutionFwdAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
}
