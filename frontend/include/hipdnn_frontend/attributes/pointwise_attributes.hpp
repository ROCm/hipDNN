// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "tensor_attributes.hpp"
#include <hipdnn_frontend/types.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class PointwiseAttributes : public AttributesCRTP<PointwiseAttributes>
{
public:
    PointwiseMode_t get_mode() const
    {
        return _mode;
    }
    std::optional<float> get_relu_lower_clip() const
    {
        return _reluLowerClip;
    }
    std::optional<float> get_relu_upper_clip() const
    {
        return _reluUpperClip;
    }
    std::optional<float> get_relu_lower_slope() const
    {
        return _reluLowerSlope;
    }
    std::optional<int64_t> get_axis() const
    {
        return _axis;
    }
    std::shared_ptr<TensorAttributes> get_input_0() const
    {
        return getInput(input_names::in_0);
    }
    std::shared_ptr<TensorAttributes> get_input_1() const
    {
        return getInput(input_names::in_1);
    }
    std::shared_ptr<TensorAttributes> get_input_2() const
    {
        return getInput(input_names::in_2);
    }
    std::shared_ptr<TensorAttributes> get_output_0() const
    {
        return getOutput(output_names::out_0);
    }

    PointwiseAttributes& set_mode(PointwiseMode_t mode)
    {
        _mode = mode;
        return *this;
    }
    PointwiseAttributes& set_relu_lower_clip(std::optional<float> reluLowerClip)
    {
        _reluLowerClip = reluLowerClip;
        return *this;
    }
    PointwiseAttributes& set_relu_upper_clip(std::optional<float> reluUpperClip)
    {
        _reluUpperClip = reluUpperClip;
        return *this;
    }
    PointwiseAttributes& set_relu_lower_clip_slope(std::optional<float> reluLowerSlope)
    {
        _reluLowerSlope = reluLowerSlope;
        return *this;
    }
    PointwiseAttributes& set_axis(std::optional<int64_t> axis)
    {
        _axis = axis;
        return *this;
    }
    PointwiseAttributes& set_input_0(const std::shared_ptr<TensorAttributes>& input0)
    {
        inputs[input_names::in_0] = input0;
        return *this;
    }
    PointwiseAttributes& set_input_0(std::shared_ptr<TensorAttributes>&& input0)
    {
        inputs[input_names::in_0] = std::move(input0);
        return *this;
    }
    PointwiseAttributes& set_input_1(const std::shared_ptr<TensorAttributes>& input1)
    {
        inputs[input_names::in_1] = input1;
        return *this;
    }
    PointwiseAttributes& set_input_1(std::shared_ptr<TensorAttributes>&& input1)
    {
        inputs[input_names::in_1] = std::move(input1);
        return *this;
    }
    PointwiseAttributes& set_input_2(const std::shared_ptr<TensorAttributes>& input2)
    {
        inputs[input_names::in_2] = input2;
        return *this;
    }
    PointwiseAttributes& set_input_2(std::shared_ptr<TensorAttributes>&& input2)
    {
        inputs[input_names::in_2] = std::move(input2);
        return *this;
    }
    PointwiseAttributes& set_output_0(const std::shared_ptr<TensorAttributes>& output0)
    {
        outputs[output_names::out_0] = output0;
        return *this;
    }
    PointwiseAttributes& set_output_0(std::shared_ptr<TensorAttributes>&& output0)
    {
        outputs[output_names::out_0] = std::move(output0);
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
    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs;

    flatbuffers::Offset<hipdnn_sdk::data_objects::PointwiseAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto in0 = get_input_0();
        auto in1 = get_input_1();
        auto in2 = get_input_2();
        auto ot0 = get_output_0();

        return hipdnn_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            to_sdk_type(_mode),
            _reluLowerClip,
            _reluUpperClip,
            _reluLowerSlope,
            _axis,
            in0->get_uid(),
            in1 ? flatbuffers::Optional<int64_t>(in1->get_uid()) : flatbuffers::nullopt,
            in2 ? flatbuffers::Optional<int64_t>(in2->get_uid()) : flatbuffers::nullopt,
            ot0->get_uid());
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
    PointwiseMode_t _mode = PointwiseMode_t::NOT_SET;
    std::optional<float> _reluLowerClip = std::nullopt;
    std::optional<float> _reluUpperClip = std::nullopt;
    std::optional<float> _reluLowerSlope = std::nullopt;
    std::optional<int64_t> _axis = std::nullopt;
};
typedef PointwiseAttributes Pointwise_attributes;
}
}
