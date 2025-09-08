// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class BatchnormInferenceAttributes : public Attributes<BatchnormInferenceAttributes>
{
public:
    enum class input_names // NOLINT(readability-identifier-naming)
    {
        X = 0,
        MEAN = 1,
        INV_VARIANCE = 2,
        SCALE = 3,
        BIAS = 4
    };

    enum class output_names // NOLINT(readability-identifier-naming)
    {
        Y = 0
    };

    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(input_names::X);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(input_names::MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getInput(input_names::INV_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(input_names::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(input_names::BIAS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(output_names::Y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::INV_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::Y, std::move(value));
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto mean = get_mean();
        auto invVariance = get_inv_variance();

        return hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
            builder,
            get_x()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            invVariance ? flatbuffers::Optional<int64_t>(invVariance->get_uid())
                        : flatbuffers::nullopt,
            get_scale()->get_uid(),
            get_bias()->get_uid(),
            get_y()->get_uid());
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

    BatchnormInferenceAttributes& setInput(input_names name,
                                           const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    BatchnormInferenceAttributes& setInput(input_names name,
                                           std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    BatchnormInferenceAttributes& setOutput(output_names name,
                                            const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    BatchnormInferenceAttributes& setOutput(output_names name,
                                            std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};
typedef BatchnormInferenceAttributes Batchnorm_inference_attributes;
}
}
