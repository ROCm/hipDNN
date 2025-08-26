// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <optional>

#include <hipdnn_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

#include "miopen_tensor.hpp"
#include "plan_interface.hpp"

namespace miopen_legacy_plugin
{

class BatchnormBwdParams
{
public:
    BatchnormBwdParams(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    const Miopen_tensor& x() const;
    const Miopen_tensor& dy() const;
    const Miopen_tensor& dx() const;
    const Miopen_tensor& scale() const;
    const Miopen_tensor& dscale() const;
    const Miopen_tensor& dbias() const;

    const std::optional<std::unique_ptr<Miopen_tensor>>& optMean() const;
    const std::optional<std::unique_ptr<Miopen_tensor>>& optInvVariance() const;

private:
    void initializeTensors(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    std::unique_ptr<Miopen_tensor> _x;
    std::unique_ptr<Miopen_tensor> _dy;
    std::unique_ptr<Miopen_tensor> _dx;
    std::unique_ptr<Miopen_tensor> _scale;
    std::unique_ptr<Miopen_tensor> _dscale;
    std::unique_ptr<Miopen_tensor> _dbias;

    std::optional<std::unique_ptr<Miopen_tensor>> _optMean;
    std::optional<std::unique_ptr<Miopen_tensor>> _optInvVariance;
};

class BatchnormBwdPlan : public PlanInterface
{
public:
    BatchnormBwdPlan(std::unique_ptr<BatchnormBwdParams> params);

    void execute(const hipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<BatchnormBwdParams> _params;
};

} // namespace miopen_legacy_plugin
