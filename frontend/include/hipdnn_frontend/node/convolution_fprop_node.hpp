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
    Conv_fprop_attributes attributes;

    ConvolutionNode(Conv_fprop_attributes&& conv_attrs, const Graph_attributes& graph_attrs)
        : NodeCRTP(graph_attrs)
        , attributes(std::move(conv_attrs))
    {
    }

    error_t pre_validate_node() const override
    {
        // Validate tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               error_code_t::ATTRIBUTE_NOT_SET,
                               "ConvolutionNode missing x (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_w(),
                               error_code_t::ATTRIBUTE_NOT_SET,
                               "ConvolutionNode missing w (weights) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               error_code_t::ATTRIBUTE_NOT_SET,
                               "ConvolutionNode missing y (output) for pre-validation");

        // Validate convolution parameters
        HIPDNN_RETURN_IF_TRUE(attributes.get_pre_padding().empty(),
                              error_code_t::ATTRIBUTE_NOT_SET,
                              "ConvolutionNode missing pre_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_post_padding().empty(),
                              error_code_t::ATTRIBUTE_NOT_SET,
                              "ConvolutionNode missing post_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_stride().empty(),
                              error_code_t::ATTRIBUTE_NOT_SET,
                              "ConvolutionNode missing stride for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_dilation().empty(),
                              error_code_t::ATTRIBUTE_NOT_SET,
                              "ConvolutionNode missing dilation for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto w = attributes.get_w();
        auto y = attributes.get_y();

        // Validate input tensor dimensions and strides
        auto& x_dims = x->get_dim();

        HIPDNN_RETURN_IF_FALSE(
            x->validate_dims_and_strides_set_and_positive(),
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Input tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_LT(
            x_dims.size(),
            3,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Input tensor must have at least 3 dimensions (N, C, spatial)");

        // Validate weight tensor dimensions and strides
        auto& w_dims = w->get_dim();

        HIPDNN_RETURN_IF_FALSE(
            w->validate_dims_and_strides_set_and_positive(),
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Weight tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_NE(
            w_dims.size(),
            x_dims.size(),
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Weight tensor dimension count must match input tensor "
            "dimension count");

        // Validate input channels match between input and weight tensors
        // For regular convolution: x_dims[1] == w_dims[1]
        // For grouped convolution: x_dims[1] % w_dims[1] == 0
        HIPDNN_RETURN_IF_NE(x_dims[1] % w_dims[1],
                            0,
                            error_code_t::INVALID_VALUE,
                            "ConvolutionNode: Input tensor channels must match weight tensor input "
                            "channels or be divisible by them for grouped convolution");

        // For grouped convolution: x_dims[1] / w_dims[1] is group count.
        // Output channels must be divisible by group count.
        auto group_count = x_dims[1] / w_dims[1];
        HIPDNN_RETURN_IF_NE(w_dims[0] % group_count,
                            0,
                            error_code_t::INVALID_VALUE,
                            "ConvolutionNode: Weight tensor output channels must be divisible by "
                            "the number of groups");

        // Validate output tensor dimensions and strides if they are set
        auto& y_dims = y->get_dim();
        auto& y_strides = y->get_stride();

        if(!y_dims.empty())
        {
            HIPDNN_RETURN_IF_NE(
                y_dims.size(),
                x_dims.size(),
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor dimension count must match input tensor "
                "dimension count");

            // Validate batch size matches
            HIPDNN_RETURN_IF_NE(
                y_dims[0],
                x_dims[0],
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor batch size must match input tensor batch size");

            // Validate output channels match weight output channels
            HIPDNN_RETURN_IF_NE(
                y_dims[1],
                w_dims[0],
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor channels must match weight tensor output channels");

            HIPDNN_RETURN_IF_FALSE(
                y->validate_dims_set_and_positive(),
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor dimensions must be set and positive");
        }

        if(!y_strides.empty())
        {
            HIPDNN_RETURN_IF_FALSE(
                y->validate_dims_and_strides_set_and_positive(),
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor dimensions and strides must be set and positive");
        }

        // Validate spatial parameter counts match spatial dimensions
        auto spatial_dims = x_dims.size() - 2; // Skip N and C dimensions
        auto& pre_padding = attributes.get_pre_padding();
        auto& post_padding = attributes.get_post_padding();
        auto& stride = attributes.get_stride();
        auto& dilation = attributes.get_dilation();

        HIPDNN_RETURN_IF_NE(
            pre_padding.size(),
            spatial_dims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: pre_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            post_padding.size(),
            spatial_dims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: post_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            stride.size(),
            spatial_dims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: stride parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            dilation.size(),
            spatial_dims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: dilation parameter count must match spatial dimension count");

        // Check spatial parameters for each dimension
        for(size_t i = 0; i < spatial_dims; ++i)
        {
            auto pre_pad = pre_padding[i];
            auto post_pad = post_padding[i];
            auto stride_val = stride[i];
            auto dilation_val = dilation[i];

            // Validate parameters
            HIPDNN_RETURN_IF_LT(
                stride_val, 1, error_code_t::INVALID_VALUE, "ConvolutionNode: Stride must be > 0");

            HIPDNN_RETURN_IF_LT(
                dilation_val, 1, error_code_t::INVALID_VALUE, "ConvolutionNode: Dilation must > 0");

            HIPDNN_RETURN_IF_LT(pre_pad,
                                0,
                                error_code_t::INVALID_VALUE,
                                "ConvolutionNode: Pre-padding must be non-negative");

            HIPDNN_RETURN_IF_LT(post_pad,
                                0,
                                error_code_t::INVALID_VALUE,
                                "ConvolutionNode: Post-padding must be non-negative");
        }

        return {};
    }

    error_t infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto w = attributes.get_w();
        auto y = attributes.get_y();

        HIPDNN_RETURN_IF_FALSE(
            x, error_code_t::ATTRIBUTE_NOT_SET, "ConvolutionNode missing x for setting properties");

        HIPDNN_RETURN_IF_FALSE(
            w, error_code_t::ATTRIBUTE_NOT_SET, "ConvolutionNode missing w for setting properties");

        HIPDNN_RETURN_IF_FALSE(
            y, error_code_t::ATTRIBUTE_NOT_SET, "ConvolutionNode missing y for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_graph_attributes(graph_attributes));

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
            y_dims[1] = w_dims[0]; // C (output channels)

            // Calculate spatial dimensions (Optional D, H, W)
            // Starting from dim 2 (skip N and C)
            for(size_t i = 2; i < x_dims.size(); ++i)
            {
                auto spatial_idx = i - 2; // Index into spatial dimension arrays

                HIPDNN_RETURN_IF_TRUE(
                    spatial_idx >= pre_padding.size() || spatial_idx >= post_padding.size()
                        || spatial_idx >= stride.size() || spatial_idx >= dilation.size(),
                    error_code_t::INVALID_VALUE,
                    "ConvolutionNode: Insufficient padding/stride/dilation parameters for spatial "
                    "dimensions");

                // Standard convolution output size formula:
                // output_size = floor((input_size + pre_padding + post_padding - dilated_kernel_size) / stride) + 1
                // where dilated_kernel_size = dilation * (kernel_size - 1) + 1

                auto input_size = x_dims[i];
                auto kernel_size = w_dims[i];
                auto pre_pad = pre_padding[spatial_idx];
                auto post_pad = post_padding[spatial_idx];
                auto stride_val = stride[spatial_idx];
                auto dilation_val = dilation[spatial_idx];

                // Validate parameters
                HIPDNN_RETURN_IF_LT(stride_val,
                                    1,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Stride must be positive");

                HIPDNN_RETURN_IF_LT(dilation_val,
                                    1,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Dilation must be positive");

                HIPDNN_RETURN_IF_LT(pre_pad,
                                    0,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Pre-padding must be non-negative");

                HIPDNN_RETURN_IF_LT(post_pad,
                                    0,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Post-padding must be non-negative");

                // Calculate dilated kernel size
                auto dilated_kernel_size = (dilation_val * (kernel_size - 1)) + 1;

                // Calculate output dimension
                auto numerator = input_size + pre_pad + post_pad - dilated_kernel_size;
                HIPDNN_RETURN_IF_LT(
                    numerator,
                    0,
                    error_code_t::INVALID_VALUE,
                    "ConvolutionNode: Invalid convolution parameters result in negative "
                    "output size");

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

            HIPDNN_RETURN_IF_TRUE(
                x_strides.empty(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "ConvolutionNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                y_dims_final.empty(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "ConvolutionNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(
                x_strides.size(),
                y_dims_final.size(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "ConvolutionNode: Stride dimension mismatch between input and output tensors");

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
