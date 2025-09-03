// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFwdAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtils.hpp>

namespace hipdnn_frontend::graph
{
class ConvolutionNode : public NodeCRTP<ConvolutionNode> //NOLINT
{
public:
    ConvFpropAttributes attributes;

    ConvolutionNode(ConvFpropAttributes&& convAttrs, const GraphAttributes& graphAttrs)
        : NodeCRTP(graphAttrs)
        , attributes(std::move(convAttrs))
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
        auto& xDims = x->get_dim();

        HIPDNN_RETURN_IF_FALSE(
            x->validate_dims_and_strides_set_and_positive(),
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Input tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_LT(
            xDims.size(),
            3,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Input tensor must have at least 3 dimensions (N, C, spatial)");

        // Validate weight tensor dimensions and strides
        auto& wDims = w->get_dim();

        HIPDNN_RETURN_IF_FALSE(
            w->validate_dims_and_strides_set_and_positive(),
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Weight tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_NE(
            wDims.size(),
            xDims.size(),
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: Weight tensor dimension count must match input tensor "
            "dimension count");

        // Validate input channels match between input and weight tensors
        // For regular convolution: x_dims[1] == w_dims[1]
        // For grouped convolution: x_dims[1] % w_dims[1] == 0
        HIPDNN_RETURN_IF_NE(xDims[1] % wDims[1],
                            0,
                            error_code_t::INVALID_VALUE,
                            "ConvolutionNode: Input tensor channels must match weight tensor input "
                            "channels or be divisible by them for grouped convolution");

        // For grouped convolution: x_dims[1] / w_dims[1] is group count.
        // Output channels must be divisible by group count.
        auto groupCount = xDims[1] / wDims[1];
        HIPDNN_RETURN_IF_NE(wDims[0] % groupCount,
                            0,
                            error_code_t::INVALID_VALUE,
                            "ConvolutionNode: Weight tensor output channels must be divisible by "
                            "the number of groups");

        // Validate output tensor dimensions and strides if they are set
        auto& yDims = y->get_dim();
        auto& yStrides = y->get_stride();

        if(!yDims.empty())
        {
            HIPDNN_RETURN_IF_NE(
                yDims.size(),
                xDims.size(),
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor dimension count must match input tensor "
                "dimension count");

            // Validate batch size matches
            HIPDNN_RETURN_IF_NE(
                yDims[0],
                xDims[0],
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor batch size must match input tensor batch size");

            // Validate output channels match weight output channels
            HIPDNN_RETURN_IF_NE(
                yDims[1],
                wDims[0],
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor channels must match weight tensor output channels");

            HIPDNN_RETURN_IF_FALSE(
                y->validate_dims_set_and_positive(),
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor dimensions must be set and positive");
        }

        if(!yStrides.empty())
        {
            HIPDNN_RETURN_IF_FALSE(
                y->validate_dims_and_strides_set_and_positive(),
                error_code_t::INVALID_VALUE,
                "ConvolutionNode: Output tensor dimensions and strides must be set and positive");
        }

        // Validate spatial parameter counts match spatial dimensions
        auto spatialDims = xDims.size() - 2; // Skip N and C dimensions
        auto& prePadding = attributes.get_pre_padding();
        auto& postPadding = attributes.get_post_padding();
        auto& stride = attributes.get_stride();
        auto& dilation = attributes.get_dilation();

        HIPDNN_RETURN_IF_NE(
            prePadding.size(),
            spatialDims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: pre_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            postPadding.size(),
            spatialDims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: post_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            stride.size(),
            spatialDims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: stride parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            dilation.size(),
            spatialDims,
            error_code_t::INVALID_VALUE,
            "ConvolutionNode: dilation parameter count must match spatial dimension count");

        // Check spatial parameters for each dimension
        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto prePad = prePadding[i];
            auto postPad = postPadding[i];
            auto strideVal = stride[i];
            auto dilationVal = dilation[i];

            // Validate parameters
            HIPDNN_RETURN_IF_LT(
                strideVal, 1, error_code_t::INVALID_VALUE, "ConvolutionNode: Stride must be > 0");

            HIPDNN_RETURN_IF_LT(
                dilationVal, 1, error_code_t::INVALID_VALUE, "ConvolutionNode: Dilation must > 0");

            HIPDNN_RETURN_IF_LT(prePad,
                                0,
                                error_code_t::INVALID_VALUE,
                                "ConvolutionNode: Pre-padding must be non-negative");

            HIPDNN_RETURN_IF_LT(postPad,
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

        auto yDims = y->get_dim();

        // Infer output dimensions if not set
        if(yDims.empty())
        {
            auto& xDims = x->get_dim();
            auto& wDims = w->get_dim();

            yDims.resize(xDims.size());

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& stride = attributes.get_stride();
            auto& dilation = attributes.get_dilation();

            yDims[0] = xDims[0]; // N (batch) matches input
            yDims[1] = wDims[0]; // C (output channels)

            // Calculate spatial dimensions (Optional D, H, W)
            // Starting from dim 2 (skip N and C)
            for(size_t i = 2; i < xDims.size(); ++i)
            {
                auto spatialIdx = i - 2; // Index into spatial dimension arrays

                HIPDNN_RETURN_IF_TRUE(
                    spatialIdx >= prePadding.size() || spatialIdx >= postPadding.size()
                        || spatialIdx >= stride.size() || spatialIdx >= dilation.size(),
                    error_code_t::INVALID_VALUE,
                    "ConvolutionNode: Insufficient padding/stride/dilation parameters for spatial "
                    "dimensions");

                // Standard convolution output size formula:
                // output_size = floor((input_size + pre_padding + post_padding - dilated_kernel_size) / stride) + 1
                // where dilated_kernel_size = dilation * (kernel_size - 1) + 1

                auto inputSize = xDims[i];
                auto kernelSize = wDims[i];
                auto prePad = prePadding[spatialIdx];
                auto postPad = postPadding[spatialIdx];
                auto strideVal = stride[spatialIdx];
                auto dilationVal = dilation[spatialIdx];

                // Validate parameters
                HIPDNN_RETURN_IF_LT(strideVal,
                                    1,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Stride must be positive");

                HIPDNN_RETURN_IF_LT(dilationVal,
                                    1,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Dilation must be positive");

                HIPDNN_RETURN_IF_LT(prePad,
                                    0,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Pre-padding must be non-negative");

                HIPDNN_RETURN_IF_LT(postPad,
                                    0,
                                    error_code_t::INVALID_VALUE,
                                    "ConvolutionNode: Post-padding must be non-negative");

                // Calculate dilated kernel size
                auto dilatedKernelSize = (dilationVal * (kernelSize - 1)) + 1;

                // Calculate output dimension
                auto numerator = inputSize + prePad + postPad - dilatedKernelSize;
                HIPDNN_RETURN_IF_LT(
                    numerator,
                    0,
                    error_code_t::INVALID_VALUE,
                    "ConvolutionNode: Invalid convolution parameters result in negative "
                    "output size");

                yDims[i] = (numerator / strideVal) + 1;
            }

            // Set the inferred dimensions
            y->set_dim(yDims);
        }

        // Infer output strides if not set
        if(y->get_stride().empty())
        {
            auto& xStrides = x->get_stride();
            auto& yDimsFinal = y->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                xStrides.empty(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "ConvolutionNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                yDimsFinal.empty(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "ConvolutionNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(
                xStrides.size(),
                yDimsFinal.size(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "ConvolutionNode: Stride dimension mismatch between input and output tensors");

            // All validations passed - perform stride generation
            std::vector<int64_t> strideOrder(xStrides.size());
            std::vector<size_t> indices(xStrides.size());
            std::iota(indices.begin(), indices.end(), 0);

            // Sort indices by their corresponding stride values (ascending)
            std::ranges::sort(indices.begin(), indices.end(), [&xStrides](size_t a, size_t b) {
                return xStrides[a] < xStrides[b];
            });

            // Assign order based on sorted indices
            for(size_t i = 0; i < indices.size(); ++i)
            {
                strideOrder[indices[i]] = static_cast<int64_t>(i);
            }

            // Generate Y strides using the extracted stride order and Y dimensions
            auto yStrides = hipdnn_sdk::utilities::generateStrides(yDimsFinal, strideOrder);

            y->set_stride(yStrides);
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::NodeAttributes_ConvolutionFwdAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
}
