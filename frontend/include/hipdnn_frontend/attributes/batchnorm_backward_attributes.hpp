// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "tensor_attributes.hpp"
#include <hipdnn_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class BatchnormBackwardAttributes : public AttributesCRTP<BatchnormBackwardAttributes>
{
public:
    enum class input_names
    {
        dy = 0,
        x = 1,
        scale = 2,
        mean = 3,
        inv_variance = 4
    };

    enum class output_names
    {
        dx = 0,
        dscale = 1,
        dbias = 2
    };

    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs;
    std::vector<std::shared_ptr<TensorAttributes>> peer_stats;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(input_names::dy);
    }
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(input_names::x);
    }
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(input_names::scale);
    }
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(input_names::mean);
    }
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getInput(input_names::inv_variance);
    }
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(output_names::dx);
    }
    std::shared_ptr<TensorAttributes> get_dscale() const
    {
        return getOutput(output_names::dscale);
    }
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(output_names::dbias);
    }
    const std::vector<std::shared_ptr<TensorAttributes>>& get_peer_stats() const
    {
        return peer_stats;
    }

    BatchnormBackwardAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::dy, value);
    }
    BatchnormBackwardAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::dy, std::move(value));
    }
    BatchnormBackwardAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::x, value);
    }
    BatchnormBackwardAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::x, std::move(value));
    }
    BatchnormBackwardAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::scale, value);
    }
    BatchnormBackwardAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::scale, std::move(value));
    }
    BatchnormBackwardAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::mean, value);
    }
    BatchnormBackwardAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::mean, std::move(value));
    }
    BatchnormBackwardAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(input_names::inv_variance, value);
    }
    BatchnormBackwardAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(input_names::inv_variance, std::move(value));
    }
    BatchnormBackwardAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::dx, value);
    }
    BatchnormBackwardAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::dx, std::move(value));
    }
    BatchnormBackwardAttributes& set_dscale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::dscale, value);
    }
    BatchnormBackwardAttributes& set_dscale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::dscale, std::move(value));
    }
    BatchnormBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(output_names::dbias, value);
    }
    BatchnormBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(output_names::dbias, std::move(value));
    }
    BatchnormBackwardAttributes&
        set_peer_stats(const std::vector<std::shared_ptr<TensorAttributes>>& value)
    {
        peer_stats = value;
        return *this;
    }
    BatchnormBackwardAttributes&
        set_peer_stats(std::vector<std::shared_ptr<TensorAttributes>>&& value)
    {
        peer_stats = std::move(value);
        return *this;
    }
    BatchnormBackwardAttributes&
        set_saved_mean_and_inv_variance(const std::shared_ptr<TensorAttributes>& mean,
                                        const std::shared_ptr<TensorAttributes>& invVariance)
    {
        return set_mean(mean).set_inv_variance(invVariance);
    }
    BatchnormBackwardAttributes&
        set_saved_mean_and_inv_variance(std::shared_ptr<TensorAttributes>&& mean,
                                        std::shared_ptr<TensorAttributes>&& invVariance)
    {
        return set_mean(std::move(mean)).set_inv_variance(std::move(invVariance));
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::BatchnormBackwardAttributes>
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

        auto mean = get_mean();
        auto invVariance = get_inv_variance();

        return hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributesDirect(
            builder,
            get_dy()->get_uid(),
            get_x()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            invVariance ? flatbuffers::Optional<int64_t>(invVariance->get_uid())
                        : flatbuffers::nullopt,
            get_scale()->get_uid(),
            &peerStatsVector,
            get_dx()->get_uid(),
            get_dscale()->get_uid(),
            get_dbias()->get_uid());
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

    BatchnormBackwardAttributes& setInput(input_names name,
                                          const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    BatchnormBackwardAttributes& setInput(input_names name,
                                          std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    BatchnormBackwardAttributes& setOutput(output_names name,
                                           const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    BatchnormBackwardAttributes& setOutput(output_names name,
                                           std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};
typedef BatchnormBackwardAttributes Batchnorm_backward_attributes;
}
}
