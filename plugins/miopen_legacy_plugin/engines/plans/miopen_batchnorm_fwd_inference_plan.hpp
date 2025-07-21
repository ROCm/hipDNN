// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "miopen_tensor.hpp"
#include "miopen_utils.hpp"
#include "plan_builder_interface.hpp" //temporary
#include "plan_interface.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>

namespace miopen_legacy_plugin
{

class Batchnorm_fwd_inference_params
{
public:
    Batchnorm_fwd_inference_params(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensor_map);

    const Miopen_tensor& x() const;
    const Miopen_tensor& y() const;
    const Miopen_tensor& scale() const;
    const Miopen_tensor& bias() const;
    const std::optional<std::unique_ptr<Miopen_tensor>>& est_mean() const;
    const std::optional<std::unique_ptr<Miopen_tensor>>& est_variance() const;

private:
    void initialize_tensors(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensor_map);

    std::unique_ptr<Miopen_tensor> _x_pair;
    std::unique_ptr<Miopen_tensor> _y_pair;
    std::unique_ptr<Miopen_tensor> _scale_pair;
    std::unique_ptr<Miopen_tensor> _bias_pair;

    std::optional<std::unique_ptr<Miopen_tensor>> _est_mean_tensor_descriptor;
    std::optional<std::unique_ptr<Miopen_tensor>> _est_variance_tensor_descriptor;
};

class Batchnorm_fwd_inference_plan : public Plan_interface
{
public:
    Batchnorm_fwd_inference_plan(std::unique_ptr<Batchnorm_fwd_inference_params> inference_params);

    void execute(const hipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                 uint32_t num_device_buffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<Batchnorm_fwd_inference_params> _inference_params;
};

}