// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "tensor_attributes.hpp"
#include <hipdnn_frontend/types.hpp>
#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class ConvFpropAttributes : public AttributesCRTP<ConvFpropAttributes>
{
public:
    enum class input_names
    {
        X = 0, // Input tensor
        W = 1 // Weights/filter tensor
    };

    enum class output_names
    {
        Y = 0 // Output tensor
    };

    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs;

    // Convolution parameters
    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    ConvolutionMode_t conv_mode = ConvolutionMode_t::CROSS_CORRELATION;

    // Getters for tensors
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(input_names::X);
    }

    std::shared_ptr<TensorAttributes> get_w() const
    {
        return getInput(input_names::W);
    }

    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(output_names::Y);
    }

    // Setters for tensors
    ConvFpropAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::X, std::move(value));
    }

    ConvFpropAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::X, value);
    }

    ConvFpropAttributes& set_w(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::W, std::move(value));
    }

    ConvFpropAttributes& set_w(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::W, value);
    }

    ConvFpropAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::Y, std::move(value));
    }

    ConvFpropAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::Y, value);
    }

    ConvFpropAttributes& set_padding(std::vector<int64_t> padding)
    {
        pre_padding = padding;
        post_padding = std::move(padding);
        return *this;
    }

    ConvFpropAttributes& set_pre_padding(const std::vector<int64_t>& padding)
    {
        pre_padding = padding;
        return *this;
    }

    ConvFpropAttributes& set_pre_padding(std::vector<int64_t>&& padding)
    {
        pre_padding = std::move(padding);
        return *this;
    }

    ConvFpropAttributes& set_post_padding(const std::vector<int64_t>& padding)
    {
        post_padding = padding;
        return *this;
    }

    ConvFpropAttributes& set_post_padding(std::vector<int64_t>&& padding)
    {
        post_padding = std::move(padding);
        return *this;
    }

    ConvFpropAttributes& set_stride(const std::vector<int64_t>& strideVals)
    {
        stride = strideVals;
        return *this;
    }

    ConvFpropAttributes& set_stride(std::vector<int64_t>&& strideVals)
    {
        stride = std::move(strideVals);
        return *this;
    }

    ConvFpropAttributes& set_dilation(const std::vector<int64_t>& dilationVals)
    {
        dilation = dilationVals;
        return *this;
    }

    ConvFpropAttributes& set_dilation(std::vector<int64_t>&& dilationVals)
    {
        dilation = std::move(dilationVals);
        return *this;
    }

    ConvFpropAttributes& set_convolution_mode(ConvolutionMode_t mode)
    {
        conv_mode = mode;
        return *this;
    }

    // Getters for convolution parameters
    const std::vector<int64_t>& get_pre_padding() const
    {
        return pre_padding;
    }
    const std::vector<int64_t>& get_post_padding() const
    {
        return post_padding;
    }
    const std::vector<int64_t>& get_stride() const
    {
        return stride;
    }
    const std::vector<int64_t>& get_dilation() const
    {
        return dilation;
    }
    ConvolutionMode_t get_convolution_mode() const
    {
        return conv_mode;
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::ConvolutionFwdAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return hipdnn_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
            builder,
            get_x()->get_uid(),
            get_w()->get_uid(),
            get_y()->get_uid(),
            &pre_padding,
            &post_padding,
            &stride,
            &dilation,
            to_sdk_type(conv_mode));
    }

private:
    std::shared_ptr<TensorAttributes> getInput(input_names name) const
    {
        auto it = inputs.find(name);
        if(it != inputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    std::shared_ptr<TensorAttributes> getOutput(output_names name) const
    {
        auto it = outputs.find(name);
        if(it != outputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    ConvFpropAttributes& setInput(input_names name, const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }

    ConvFpropAttributes& setInput(input_names name, std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    ConvFpropAttributes& setOutput(output_names name,
                                   const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }

    ConvFpropAttributes& setOutput(output_names name, std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};
typedef ConvFpropAttributes Conv_fprop_attributes;
}
}
