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

class Batchnorm_bwd_params
{
public:
    Batchnorm_bwd_params(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensor_map);

    const MiopenTensor& x() const;
    const MiopenTensor& dy() const;
    const MiopenTensor& dx() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& dscale() const;
    const MiopenTensor& dbias() const;

    const std::optional<std::unique_ptr<MiopenTensor>>& opt_mean() const;
    const std::optional<std::unique_ptr<MiopenTensor>>& opt_inv_variance() const;

private:
    void initialize_tensors(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensor_map);

    std::unique_ptr<MiopenTensor> _x;
    std::unique_ptr<MiopenTensor> _dy;
    std::unique_ptr<MiopenTensor> _dx;
    std::unique_ptr<MiopenTensor> _scale;
    std::unique_ptr<MiopenTensor> _dscale;
    std::unique_ptr<MiopenTensor> _dbias;

    std::optional<std::unique_ptr<MiopenTensor>> _opt_mean;
    std::optional<std::unique_ptr<MiopenTensor>> _opt_inv_variance;
};

class Batchnorm_bwd_plan : public Plan_interface
{
public:
    Batchnorm_bwd_plan(std::unique_ptr<Batchnorm_bwd_params> params);

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                 uint32_t num_device_buffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<Batchnorm_bwd_params> _params;
};

} // namespace miopen_legacy_plugin
