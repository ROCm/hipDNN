// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "tensor_attributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class Batchnorm_inference_attributes : public AttributesCRTP<Batchnorm_inference_attributes>
{
public:
    enum class input_names
    {
        x            = 0,
        mean         = 1,
        inv_variance = 2,
        scale        = 3,
        bias         = 4
    };

    enum class output_names
    {
        y = 0
    };

    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>>  inputs;
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    std::shared_ptr<Tensor_attributes> get_x() const
    {
        return get_input(input_names::x);
    }
    std::shared_ptr<Tensor_attributes> get_mean() const
    {
        return get_input(input_names::mean);
    }
    std::shared_ptr<Tensor_attributes> get_inv_variance() const
    {
        return get_input(input_names::inv_variance);
    }
    std::shared_ptr<Tensor_attributes> get_scale() const
    {
        return get_input(input_names::scale);
    }
    std::shared_ptr<Tensor_attributes> get_bias() const
    {
        return get_input(input_names::bias);
    }
    std::shared_ptr<Tensor_attributes> get_y() const
    {
        return get_output(output_names::y);
    }

    Batchnorm_inference_attributes& set_x(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::x, value);
    }
    Batchnorm_inference_attributes& set_mean(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::mean, value);
    }
    Batchnorm_inference_attributes&
        set_inv_variance(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::inv_variance, value);
    }
    Batchnorm_inference_attributes& set_scale(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::scale, value);
    }
    Batchnorm_inference_attributes& set_bias(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::bias, value);
    }
    Batchnorm_inference_attributes& set_y(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::y, value);
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

    Batchnorm_inference_attributes& set_input(input_names                               name,
                                              const std::shared_ptr<Tensor_attributes>& value)
    {
        inputs[name] = value;
        return *this;
    }

    Batchnorm_inference_attributes& set_output(output_names                              name,
                                               const std::shared_ptr<Tensor_attributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
};
}
}
