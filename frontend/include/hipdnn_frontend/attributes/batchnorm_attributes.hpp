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
class Batchnorm_attributes : public AttributesCRTP<Batchnorm_attributes>
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

    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;

    std::shared_ptr<Tensor_attributes> get_x() const
    {
        return get_input(input_names::x);
    }
    std::shared_ptr<Tensor_attributes> get_scale() const
    {
        return get_input(input_names::scale);
    }
    std::shared_ptr<Tensor_attributes> get_bias() const
    {
        return get_input(input_names::bias);
    }
    std::shared_ptr<Tensor_attributes> get_epsilon() const
    {
        return get_input(input_names::epsilon);
    }
    const std::vector<std::shared_ptr<Tensor_attributes>>& get_peer_stats() const
    {
        return peer_stats;
    }
    std::shared_ptr<Tensor_attributes> get_prev_running_mean() const
    {
        return get_input(input_names::prev_running_mean);
    }
    std::shared_ptr<Tensor_attributes> get_prev_running_variance() const
    {
        return get_input(input_names::prev_running_variance);
    }
    std::shared_ptr<Tensor_attributes> get_momentum() const
    {
        return get_input(input_names::momentum);
    }
    std::shared_ptr<Tensor_attributes> get_y() const
    {
        return get_output(output_names::y);
    }
    std::shared_ptr<Tensor_attributes> get_mean() const
    {
        return get_output(output_names::mean);
    }
    std::shared_ptr<Tensor_attributes> get_inv_variance() const
    {
        return get_output(output_names::inv_variance);
    }
    std::shared_ptr<Tensor_attributes> get_next_running_mean() const
    {
        return get_output(output_names::next_running_mean);
    }
    std::shared_ptr<Tensor_attributes> get_next_running_variance() const
    {
        return get_output(output_names::next_running_variance);
    }

    Batchnorm_attributes& set_x(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::x, value);
    }
    Batchnorm_attributes& set_scale(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::scale, value);
    }
    Batchnorm_attributes& set_bias(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::bias, value);
    }
    Batchnorm_attributes& set_epsilon(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::epsilon, value);
    }
    Batchnorm_attributes&
        set_peer_stats(const std::vector<std::shared_ptr<Tensor_attributes>>& value)
    {
        peer_stats = value;
        return *this;
    }
    Batchnorm_attributes& set_prev_running_mean(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::prev_running_mean, value);
    }
    Batchnorm_attributes& set_prev_running_variance(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::prev_running_variance, value);
    }
    Batchnorm_attributes& set_momentum(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::momentum, value);
    }
    Batchnorm_attributes& set_y(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::y, value);
    }
    Batchnorm_attributes& set_mean(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::mean, value);
    }
    Batchnorm_attributes& set_inv_variance(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::inv_variance, value);
    }
    Batchnorm_attributes& set_next_running_mean(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::next_running_mean, value);
    }
    Batchnorm_attributes& set_next_running_variance(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::next_running_variance, value);
    }
    Batchnorm_attributes&
        set_previous_running_stats(const std::shared_ptr<Tensor_attributes>& mean,
                                   const std::shared_ptr<Tensor_attributes>& variance,
                                   const std::shared_ptr<Tensor_attributes>& momentum)
    {
        return set_prev_running_mean(mean).set_prev_running_variance(variance).set_momentum(
            momentum);
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::BatchnormAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const
    {
        auto peer_stats_vector = std::vector<int64_t>{};
        for(const auto& peer_stat : peer_stats)
        {
            if(peer_stat)
            {
                peer_stats_vector.emplace_back(peer_stat->get_uid());
            }
        }

        auto prev_running_mean = get_prev_running_mean();
        auto prev_running_variance = get_prev_running_variance();
        auto momentum = get_momentum();
        auto mean = get_mean();
        auto inv_variance = get_inv_variance();
        auto next_running_mean = get_next_running_mean();
        auto next_running_variance = get_next_running_variance();

        return hipdnn_sdk::data_objects::CreateBatchnormAttributesDirect(
            builder,
            get_x()->get_uid(),
            get_scale()->get_uid(),
            get_bias()->get_uid(),
            get_epsilon()->get_uid(),
            &peer_stats_vector,
            prev_running_mean ? flatbuffers::Optional<int64_t>(prev_running_mean->get_uid())
                              : flatbuffers::nullopt,
            prev_running_variance ? flatbuffers::Optional<int64_t>(prev_running_variance->get_uid())
                                  : flatbuffers::nullopt,
            momentum ? flatbuffers::Optional<int64_t>(momentum->get_uid()) : flatbuffers::nullopt,
            get_y()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            inv_variance ? flatbuffers::Optional<int64_t>(inv_variance->get_uid())
                         : flatbuffers::nullopt,
            next_running_mean ? flatbuffers::Optional<int64_t>(next_running_mean->get_uid())
                              : flatbuffers::nullopt,
            next_running_variance ? flatbuffers::Optional<int64_t>(next_running_variance->get_uid())
                                  : flatbuffers::nullopt);
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

    Batchnorm_attributes& set_input(input_names name,
                                    const std::shared_ptr<Tensor_attributes>& value)
    {
        inputs[name] = value;
        return *this;
    }

    Batchnorm_attributes& set_output(output_names name,
                                     const std::shared_ptr<Tensor_attributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
};
}
}
