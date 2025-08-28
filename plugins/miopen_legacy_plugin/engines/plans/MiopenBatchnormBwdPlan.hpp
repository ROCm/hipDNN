// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <optional>

#include <hipdnn_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

#include "MiopenTensor.hpp"
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class BatchnormBwdParams
{
public:
    BatchnormBwdParams(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    const MiopenTensor& x() const;
    const MiopenTensor& dy() const;
    const MiopenTensor& dx() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& dscale() const;
    const MiopenTensor& dbias() const;

    const std::optional<std::unique_ptr<MiopenTensor>>& optMean() const;
    const std::optional<std::unique_ptr<MiopenTensor>>& optInvVariance() const;

private:
    void initializeTensors(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    std::unique_ptr<MiopenTensor> _x;
    std::unique_ptr<MiopenTensor> _dy;
    std::unique_ptr<MiopenTensor> _dx;
    std::unique_ptr<MiopenTensor> _scale;
    std::unique_ptr<MiopenTensor> _dscale;
    std::unique_ptr<MiopenTensor> _dbias;

    std::optional<std::unique_ptr<MiopenTensor>> _optMean;
    std::optional<std::unique_ptr<MiopenTensor>> _optInvVariance;
};

class BatchnormBwdPlan : public IPlan
{
public:
    BatchnormBwdPlan(std::unique_ptr<BatchnormBwdParams> params);

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<BatchnormBwdParams> _params;
};

} // namespace miopen_legacy_plugin
