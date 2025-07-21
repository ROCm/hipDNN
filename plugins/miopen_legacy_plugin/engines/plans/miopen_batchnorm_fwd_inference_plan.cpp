// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_batchnorm_fwd_inference_plan.hpp"

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

Batchnorm_fwd_inference_params::Batchnorm_fwd_inference_params(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensor_map)
{
    initialize_tensors(attributes, tensor_map);
}

const Miopen_tensor& Batchnorm_fwd_inference_params::x() const
{
    return *_x_pair;
}

const Miopen_tensor& Batchnorm_fwd_inference_params::y() const
{
    return *_y_pair;
}

const Miopen_tensor& Batchnorm_fwd_inference_params::scale() const
{
    return *_scale_pair;
}

const Miopen_tensor& Batchnorm_fwd_inference_params::bias() const
{
    return *_bias_pair;
}

const std::optional<std::unique_ptr<Miopen_tensor>>&
    Batchnorm_fwd_inference_params::est_mean() const
{
    return _est_mean_tensor_descriptor;
}

const std::optional<std::unique_ptr<Miopen_tensor>>&
    Batchnorm_fwd_inference_params::est_variance() const
{
    return _est_variance_tensor_descriptor;
}

std::unique_ptr<Miopen_tensor> create_tensor(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensor_map,
    int64_t uid)
{
    if(auto tensor_attr = tensor_map.find(uid); tensor_attr != tensor_map.end())
    {
        return std::make_unique<Miopen_tensor>(*tensor_attr->second);
    }

    throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                 "Failed to find tensor with UID in tensor_map: "
                                                     + std::to_string(uid));
}

void Batchnorm_fwd_inference_params::initialize_tensors(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensor_map)
{
    _x_pair = create_tensor(tensor_map, attributes.x());
    _y_pair = create_tensor(tensor_map, attributes.y());
    _scale_pair = create_tensor(tensor_map, attributes.scale());
    _bias_pair = create_tensor(tensor_map, attributes.bias());

    if(attributes.mean().has_value())
    {
        _est_mean_tensor_descriptor = create_tensor(tensor_map, attributes.mean().value());
    }
    if(attributes.inv_variance().has_value())
    {
        _est_variance_tensor_descriptor
            = create_tensor(tensor_map, attributes.inv_variance().value());
    }
}

Batchnorm_fwd_inference_plan::Batchnorm_fwd_inference_plan(
    std::unique_ptr<Batchnorm_fwd_inference_params> inference_params)
    : _inference_params(std::move(inference_params))
{
}

void Batchnorm_fwd_inference_plan::execute(const hipdnnEnginePluginHandle& handle,
                                           const hipdnnPluginDeviceBuffer_t* device_buffers,
                                           uint32_t num_device_buffers,
                                           void* workspace) const
{
    std::ignore = workspace;

    // Hardcoded values from bn_driver in miopen
    float alpha = static_cast<float>(1);
    float beta = static_cast<float>(0);
    double epsilon = 1e-3;

    auto x_buffer = miopen_utils::find_device_buffer(
        _inference_params->x().uid(), device_buffers, num_device_buffers);
    auto y_buffer = miopen_utils::find_device_buffer(
        _inference_params->y().uid(), device_buffers, num_device_buffers);
    auto scale_buffer = miopen_utils::find_device_buffer(
        _inference_params->scale().uid(), device_buffers, num_device_buffers);
    auto bias_buffer = miopen_utils::find_device_buffer(
        _inference_params->bias().uid(), device_buffers, num_device_buffers);

    hipdnnPluginDeviceBuffer_t est_mean_buffer = {0, nullptr};
    if(_inference_params->est_mean().has_value())
    {
        est_mean_buffer = miopen_utils::find_device_buffer(
            _inference_params->est_mean().value()->uid(), device_buffers, num_device_buffers);
    }

    hipdnnPluginDeviceBuffer_t est_variance_buffer = {0, nullptr};
    if(_inference_params->est_variance().has_value())
    {
        est_variance_buffer = miopen_utils::find_device_buffer(
            _inference_params->est_variance().value()->uid(), device_buffers, num_device_buffers);
    }

    THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationForwardInference_V2(
        handle.miopen_handle,
        MIOPEN_BATCHNORM_MODE,
        &alpha,
        &beta,
        _inference_params->x().tensor_descriptor(),
        x_buffer.ptr,
        _inference_params->y().tensor_descriptor(),
        y_buffer.ptr,
        _inference_params->scale().tensor_descriptor(),
        _inference_params->bias().tensor_descriptor(),
        _inference_params->est_mean().has_value()
            ? _inference_params->est_mean().value()->tensor_descriptor()
            : nullptr,
        _inference_params->est_variance().has_value()
            ? _inference_params->est_variance().value()->tensor_descriptor()
            : nullptr,
        scale_buffer.ptr,
        bias_buffer.ptr,
        est_mean_buffer.ptr,
        est_variance_buffer.ptr,
        epsilon));
}

}