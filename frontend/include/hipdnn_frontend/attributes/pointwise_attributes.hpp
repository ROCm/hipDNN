// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "tensor_attributes.hpp"
#include "tensor_attributes_generated.h"
#include <hipdnn_frontend/types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class Pointwise_attributes : public AttributesCRTP<Pointwise_attributes>
{
public:
    PointwiseMode_t get_operation() const
    {
        return _operation;
    }
    std::optional<float> get_relu_lower_clip() const
    {
        return _relu_lower_clip;
    }
    std::optional<float> get_relu_upper_clip() const
    {
        return _relu_upper_clip;
    }
    std::optional<float> get_relu_lower_slope() const
    {
        return _relu_lower_slope;
    }
    std::optional<int64_t> get_axis() const
    {
        return _axis;
    }
    std::shared_ptr<Tensor_attributes> get_input_0() const
    {
        return get_input(input_names::in_0);
    }
    std::shared_ptr<Tensor_attributes> get_input_1() const
    {
        return get_input(input_names::in_1);
    }
    std::shared_ptr<Tensor_attributes> get_input_2() const
    {
        return get_input(input_names::in_2);
    }
    std::shared_ptr<Tensor_attributes> get_output_0() const
    {
        return get_output(output_names::out_0);
    }

    Pointwise_attributes& set_operation(PointwiseMode_t operation)
    {
        _operation = operation;
        return *this;
    }
    Pointwise_attributes& set_relu_lower_clip(std::optional<float> relu_lower_clip)
    {
        _relu_lower_clip = relu_lower_clip;
        return *this;
    }
    Pointwise_attributes& set_relu_upper_clip(std::optional<float> relu_upper_clip)
    {
        _relu_upper_clip = relu_upper_clip;
        return *this;
    }
    Pointwise_attributes& set_relu_lower_slope(std::optional<float> relu_lower_slope)
    {
        _relu_lower_slope = relu_lower_slope;
        return *this;
    }
    Pointwise_attributes& set_axis(std::optional<int64_t> axis)
    {
        _axis = axis;
        return *this;
    }
    Pointwise_attributes& set_input_0(const std::shared_ptr<Tensor_attributes>& input_0)
    {
        inputs[input_names::in_0] = input_0;
        return *this;
    }
    Pointwise_attributes& set_input_1(const std::shared_ptr<Tensor_attributes>& input_1)
    {
        inputs[input_names::in_1] = input_1;
        return *this;
    }
    Pointwise_attributes& set_input_2(const std::shared_ptr<Tensor_attributes>& input_2)
    {
        inputs[input_names::in_2] = input_2;
        return *this;
    }
    Pointwise_attributes& set_output_0(const std::shared_ptr<Tensor_attributes>& output_0)
    {
        outputs[output_names::out_0] = output_0;
        return *this;
    }

    enum class input_names
    {
        in_0 = 0,
        in_1 = 1,
        in_2 = 2,
    };
    enum class output_names
    {
        out_0 = 0,
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>>  inputs;
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    flatbuffers::Offset<hipdnn::sdk::PointwiseAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const
    {
        auto in_0  = get_input_0();
        auto in_1  = get_input_1();
        auto in_2  = get_input_2();
        auto out_0 = get_output_0();

        return hipdnn::sdk::CreatePointwiseAttributes(
            builder,
            to_sdk_type(_operation),
            _relu_lower_clip,
            _relu_upper_clip,
            _relu_lower_slope,
            _axis,
            in_0->get_uid(),
            in_1 ? flatbuffers::Optional<int64_t>(in_1->get_uid()) : flatbuffers::nullopt,
            in_2 ? flatbuffers::Optional<int64_t>(in_2->get_uid()) : flatbuffers::nullopt,
            out_0->get_uid());
    }

private:
    std::shared_ptr<Tensor_attributes> get_input(input_names name) const
    {
        auto it = inputs.find(name);
        if(it != inputs.end())
        {
            return it->second;
        }
        return nullptr;
    }
    std::shared_ptr<Tensor_attributes> get_output(output_names name) const
    {
        auto it = outputs.find(name);
        if(it != outputs.end())
        {
            return it->second;
        }
        return nullptr;
    }
    PointwiseMode_t        _operation        = PointwiseMode_t::NOT_SET;
    std::optional<float>   _relu_lower_clip  = std::nullopt;
    std::optional<float>   _relu_upper_clip  = std::nullopt;
    std::optional<float>   _relu_lower_slope = std::nullopt;
    std::optional<int64_t> _axis             = std::nullopt;
};
}
}
