// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes.hpp"
#include "batchnorm_backward_attributes_generated.h"
#include "tensor_attributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class Batchnorm_backward_attributes : public AttributesCRTP<Batchnorm_backward_attributes>
{
public:
    enum class input_names
    {
        dy           = 0,
        x            = 1,
        scale        = 2,
        mean         = 3,
        inv_variance = 4
    };

    enum class output_names
    {
        dx     = 0,
        dscale = 1,
        dbias  = 2
    };

    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>>  inputs;
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    std::vector<std::shared_ptr<Tensor_attributes>>                      peer_stats;

    std::shared_ptr<Tensor_attributes> get_dy() const
    {
        return get_input(input_names::dy);
    }
    std::shared_ptr<Tensor_attributes> get_x() const
    {
        return get_input(input_names::x);
    }
    std::shared_ptr<Tensor_attributes> get_scale() const
    {
        return get_input(input_names::scale);
    }
    std::shared_ptr<Tensor_attributes> get_mean() const
    {
        return get_input(input_names::mean);
    }
    std::shared_ptr<Tensor_attributes> get_inv_variance() const
    {
        return get_input(input_names::inv_variance);
    }
    std::shared_ptr<Tensor_attributes> get_dx() const
    {
        return get_output(output_names::dx);
    }
    std::shared_ptr<Tensor_attributes> get_dscale() const
    {
        return get_output(output_names::dscale);
    }
    std::shared_ptr<Tensor_attributes> get_dbias() const
    {
        return get_output(output_names::dbias);
    }
    const std::vector<std::shared_ptr<Tensor_attributes>>& get_peer_stats() const
    {
        return peer_stats;
    }

    Batchnorm_backward_attributes& set_dy(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::dy, value);
    }
    Batchnorm_backward_attributes& set_x(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::x, value);
    }
    Batchnorm_backward_attributes& set_scale(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::scale, value);
    }
    Batchnorm_backward_attributes& set_mean(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::mean, value);
    }
    Batchnorm_backward_attributes& set_inv_variance(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_input(input_names::inv_variance, value);
    }
    Batchnorm_backward_attributes& set_dx(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::dx, value);
    }
    Batchnorm_backward_attributes& set_dscale(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::dscale, value);
    }
    Batchnorm_backward_attributes& set_dbias(const std::shared_ptr<Tensor_attributes>& value)
    {
        return set_output(output_names::dbias, value);
    }
    Batchnorm_backward_attributes&
        set_peer_stats(const std::vector<std::shared_ptr<Tensor_attributes>>& value)
    {
        peer_stats = value;
        return *this;
    }
    Batchnorm_backward_attributes&
        set_saved_mean_and_inv_variance(const std::shared_ptr<Tensor_attributes>& mean,
                                        const std::shared_ptr<Tensor_attributes>& inv_variance)
    {

        return set_mean(mean).set_inv_variance(inv_variance);
    }

    flatbuffers::Offset<hipdnn::sdk::BatchNormBackwardAttributes>
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

        auto mean         = get_mean();
        auto inv_variance = get_inv_variance();

        return hipdnn::sdk::CreateBatchNormBackwardAttributesDirect(
            builder,
            get_dy()->get_uid(),
            get_x()->get_uid(),
            mean ? flatbuffers::Optional<int64_t>(mean->get_uid()) : flatbuffers::nullopt,
            inv_variance ? flatbuffers::Optional<int64_t>(inv_variance->get_uid())
                         : flatbuffers::nullopt,
            get_scale()->get_uid(),
            &peer_stats_vector,
            get_dx()->get_uid(),
            get_dscale()->get_uid(),
            get_dbias()->get_uid());
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

    Batchnorm_backward_attributes& set_input(input_names                               name,
                                             const std::shared_ptr<Tensor_attributes>& value)
    {
        inputs[name] = value;
        return *this;
    }

    Batchnorm_backward_attributes& set_output(output_names                              name,
                                              const std::shared_ptr<Tensor_attributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
};
}
}
