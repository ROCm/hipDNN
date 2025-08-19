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

        auto y_dims = y->get_dim();

        // Infer output dimensions if not set
        if(y_dims.empty())
        {
            auto& x_dims = x->get_dim();
            auto& w_dims = w->get_dim();

            y_dims.resize(x_dims.size());

            auto& pre_padding = attributes.get_pre_padding();
            auto& post_padding = attributes.get_post_padding();
            auto& stride = attributes.get_stride();
            auto& dilation = attributes.get_dilation();

            y_dims[0] = x_dims[0]; // N (batch) matches input
            y_dims[1] = w_dims[0]; // C (output channels) matches weight output channels

            // Calculate spatial dimensions (Optional D, H, W)
            // Starting from dim 2 (skip N and C)
            for(size_t i = 2; i < x_dims.size(); ++i)
            {
                auto spatial_idx = i - 2; // Index into spatial dimension arrays

                if(spatial_idx >= pre_padding.size() || spatial_idx >= post_padding.size()
                   || spatial_idx >= stride.size() || spatial_idx >= dilation.size())
                {
                    return {error_code_t::ATTRIBUTE_NOT_SET,
                            "ConvolutionNode: Insufficient padding/stride/dilation parameters for "
                            "spatial dimensions"};
                }

                // Standard convolution output size formula:
                // output_size = floor((input_size + pre_padding + post_padding - dilated_kernel_size) / stride) + 1
                // where dilated_kernel_size = dilation * (kernel_size - 1) + 1

                auto input_size = x_dims[i];
                auto kernel_size = w_dims[i]; // Weight spatial dimensions start from index 2
                auto pre_pad = pre_padding[spatial_idx];
                auto post_pad = post_padding[spatial_idx];
                auto stride_val = stride[spatial_idx];
                auto dilation_val = dilation[spatial_idx];

                // Validate parameters
                if(stride_val <= 0)
                {
                    return {error_code_t::ATTRIBUTE_NOT_SET,
                            "ConvolutionNode: Stride must be positive"};
                }
                if(dilation_val <= 0)
                {
                    return {error_code_t::ATTRIBUTE_NOT_SET,
                            "ConvolutionNode: Dilation must be positive"};
                }

                // Calculate dilated kernel size
                auto dilated_kernel_size = (dilation_val * (kernel_size - 1)) + 1;

                // Calculate output dimension
                auto numerator = input_size + pre_pad + post_pad - dilated_kernel_size;
                if(numerator < 0)
                {
                    return {error_code_t::ATTRIBUTE_NOT_SET,
                            "ConvolutionNode: Invalid convolution parameters result in negative "
                            "output size"};
                }

                y_dims[i] = (numerator / stride_val) + 1;
            }

            // Set the inferred dimensions
            y->set_dim(y_dims);
        }

        // Infer output strides if not set
        if(y->get_stride().empty())
        {
            auto& x_strides = x->get_stride();
            auto& y_dims_final = y->get_dim();

            // Consolidate all validation checks upfront
            if(x_strides.empty())
            {
                return {error_code_t::ATTRIBUTE_NOT_SET,
                        "ConvolutionNode: Cannot infer output strides - missing input strides"};
            }

            if(y_dims_final.empty())
            {
                return {error_code_t::ATTRIBUTE_NOT_SET,
                        "ConvolutionNode: Cannot infer output strides - missing output dimensions"};
            }

            if(x_strides.size() != y_dims_final.size())
            {
                return {
                    error_code_t::ATTRIBUTE_NOT_SET,
                    "ConvolutionNode: Stride dimension mismatch between input and output tensors"};
            }

            // All validations passed - perform stride generation
            std::vector<int64_t> stride_order(x_strides.size());
            std::vector<size_t> indices(x_strides.size());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort indices by their corresponding stride values (ascending)
            std::ranges::sort(indices.begin(), indices.end(), [&x_strides](size_t a, size_t b) {
                return x_strides[a] < x_strides[b];
            });

            // Assign order based on sorted indices
            for(size_t i = 0; i < indices.size(); ++i)
            {
                stride_order[indices[i]] = static_cast<int64_t>(i);
            }

            // Generate Y strides using the extracted stride order and Y dimensions
            auto y_strides = hipdnn_sdk::utilities::generate_strides(y_dims_final, stride_order);

            y->set_stride(y_strides);
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
