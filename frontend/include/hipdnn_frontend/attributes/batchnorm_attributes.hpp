// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "tensor_attributes.hpp"
#include <hipdnn_sdk/data_objects/batchnorm_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class BatchnormAttributes : public AttributesCRTP<BatchnormAttributes>
{
public:
    enum class input_names
    {
        x = 0,
        scale = 1,
        bias = 2,
        prev_running_mean = 3,
        prev_running_variance = 4,
        momentum = 5,
        epsilon = 6
    };

    enum class output_names
    {
        y = 0,
        mean = 1,
        inv_variance = 2,
        next_running_mean = 3,
        next_running_variance = 4
    };

    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs;
    std::vector<std::shared_ptr<TensorAttributes>> peer_stats;

    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(input_names::x);
    }
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(input_names::scale);
    }
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(input_names::bias);
    }
    std::shared_ptr<TensorAttributes> get_epsilon() const
    {
        return getInput(input_names::epsilon);
    }
    const std::vector<std::shared_ptr<TensorAttributes>>& get_peer_stats() const
    {
        return peer_stats;
    }
    std::shared_ptr<TensorAttributes> get_prev_running_mean() const
    {
        return getInput(input_names::prev_running_mean);
    }
    std::shared_ptr<TensorAttributes> get_prev_running_variance() const
    {
        return getInput(input_names::prev_running_variance);
    }
    std::shared_ptr<TensorAttributes> get_momentum() const
    {
        return getInput(input_names::momentum);
    }
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(output_names::y);
    }
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getOutput(output_names::mean);
    }
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getOutput(output_names::inv_variance);
    }
    std::shared_ptr<TensorAttributes> get_next_running_mean() const
    {
        return getOutput(output_names::next_running_mean);
    }
    std::shared_ptr<TensorAttributes> get_next_running_variance() const
    {
        return getOutput(output_names::next_running_variance);
    }
    BatchnormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::x, value);
    }
    BatchnormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::x, std::move(value));
    }
    BatchnormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::scale, value);
    }
    BatchnormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::scale, std::move(value));
    }
    BatchnormAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::bias, value);
    }
    BatchnormAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::bias, std::move(value));
    }
    BatchnormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::epsilon, value);
    }
    BatchnormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::epsilon, std::move(value));
    }

    BatchnormAttributes& set_peer_stats(const std::vector<std::shared_ptr<TensorAttributes>>& value)
    {
        peer_stats = value;
        return *this;
    }
    BatchnormAttributes& set_peer_stats(std::vector<std::shared_ptr<TensorAttributes>>&& value)
    {
        peer_stats = std::move(value);
        return *this;
    }
    BatchnormAttributes& set_prev_running_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::prev_running_mean, value);
    }
    BatchnormAttributes& set_prev_running_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::prev_running_mean, std::move(value));
    }
    BatchnormAttributes& set_prev_running_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::prev_running_variance, value);
    }
    BatchnormAttributes& set_prev_running_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::prev_running_variance, std::move(value));
    }
    BatchnormAttributes& set_momentum(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::momentum, value);
    }
    BatchnormAttributes& set_momentum(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::momentum, std::move(value));
    }
    BatchnormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::y, value);
    }
    BatchnormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::y, std::move(value));
    }
    BatchnormAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::mean, value);
    }
    BatchnormAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::mean, std::move(value));
    }
    BatchnormAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::inv_variance, value);
    }
    BatchnormAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::inv_variance, std::move(value));
    }
    BatchnormAttributes& set_next_running_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::next_running_mean, value);
    }
    BatchnormAttributes& set_next_running_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::next_running_mean, std::move(value));
    }
    BatchnormAttributes& set_next_running_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::next_running_variance, value);
    }
    BatchnormAttributes& set_next_running_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::next_running_variance, std::move(value));
    }
    BatchnormAttributes&
        set_previous_running_stats(const std::shared_ptr<TensorAttributes>& mean,
                                   const std::shared_ptr<TensorAttributes>& variance,
                                   const std::shared_ptr<TensorAttributes>& momentum)
    {
        return set_prev_running_mean(mean).set_prev_running_variance(variance).set_momentum(
            momentum);
    }
    BatchnormAttributes& set_previous_running_stats(std::shared_ptr<TensorAttributes>&& mean,
                                                    std::shared_ptr<TensorAttributes>&& variance,
                                                    std::shared_ptr<TensorAttributes>&& momentum)
    {
        return set_prev_running_mean(std::move(mean))
            .set_prev_running_variance(std::move(variance))
            .set_momentum(std::move(momentum));
    }
    flatbuffers::Offset<hipdnn_sdk::data_objects::BatchnormAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto peerStatsVector = std::vector<int64_t>{};
        for(const auto& peerStat : peer_stats)
        {
            if(peerStat)
            {
                peerStatsVector.emplace_back(peerStat->get_uid());
            }
        }

        auto prevRunningMean = get_prev_running_mean();
        auto prevRunningVariance = get_prev_running_variance();
        auto momentum = get_momentum();
        auto mean = get_mean();
        auto invVariance = get_inv_variance();
        auto nextRunningMean = get_next_running_mean();
        auto nextRunningVariance = get_next_running_variance();

        return hipdnn_sdk::data_objects::CreateBatchnormAttributesDirect(
            builder,
            get_x()->get_uid(),
            get_scale()->get_uid(),
            get_bias()->get_uid(),
            get_epsilon()->get_uid(),
            &peerStatsVector,
            prevRunningMean ? flatbuffers::Optional<int64_t>(prevRunningMean->get_uid())
                            : flatbuffers::nullopt,
            prevRunningVariance ? flatbuffers::Optional<int64_t>(prevRunningVariance->get_uid())
                                : flatbuffers::nullopt,
            momentum ? flatbuffers::Optional<int64_t>(momentum->get_uid()) : flatbuffers::nullopt,
            get_y()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            invVariance ? flatbuffers::Optional<int64_t>(invVariance->get_uid())
                        : flatbuffers::nullopt,
            nextRunningMean ? flatbuffers::Optional<int64_t>(nextRunningMean->get_uid())
                            : flatbuffers::nullopt,
            nextRunningVariance ? flatbuffers::Optional<int64_t>(nextRunningVariance->get_uid())
                                : flatbuffers::nullopt);
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

    BatchnormAttributes& setInput(input_names name, const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    BatchnormAttributes& setInput(input_names name, std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    BatchnormAttributes& setOutput(output_names name,
                                   const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    BatchnormAttributes& setOutput(output_names name, std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};

typedef BatchnormAttributes Batchnorm_attributes;
}
}
