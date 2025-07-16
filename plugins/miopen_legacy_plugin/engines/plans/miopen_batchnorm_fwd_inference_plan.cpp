// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_batchnorm_fwd_inference_plan.hpp"

namespace miopen_legacy_plugin
{

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

const std::optional<const Miopen_tensor*> Batchnorm_fwd_inference_params::est_mean() const
{
    if(_est_mean_tensor_descriptor)
    {
        return std::optional<const Miopen_tensor*>(_est_mean_tensor_descriptor.get());
    }
    return std::optional<const Miopen_tensor*>();
}

const std::optional<const Miopen_tensor*> Batchnorm_fwd_inference_params::est_variance() const
{
    if(_est_variance_tensor_descriptor)
    {
        return std::optional<const Miopen_tensor*>(_est_variance_tensor_descriptor.get());
    }
    return std::optional<const Miopen_tensor*>();
}

void Batchnorm_fwd_inference_params::initialize_tensors(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensor_map)
{
    if(auto x_tensor_attr = tensor_map.find(attributes.x()); x_tensor_attr != tensor_map.end())
    {
        _x_pair = std::make_unique<Miopen_tensor>(*x_tensor_attr->second);
    }
    if(auto y_tensor_attr = tensor_map.find(attributes.y()); y_tensor_attr != tensor_map.end())
    {
        _y_pair = std::make_unique<Miopen_tensor>(*y_tensor_attr->second);
    }
    if(auto scale_tensor_attr = tensor_map.find(attributes.scale());
       scale_tensor_attr != tensor_map.end())
    {
        _scale_pair = std::make_unique<Miopen_tensor>(*scale_tensor_attr->second);
    }
    if(auto bias_tensor_attr = tensor_map.find(attributes.bias());
       bias_tensor_attr != tensor_map.end())
    {
        _bias_pair = std::make_unique<Miopen_tensor>(*bias_tensor_attr->second);
    }

    if(attributes.mean().has_value())
    {
        if(auto est_mean_attr = tensor_map.find(attributes.mean().value());
           est_mean_attr != tensor_map.end())
        {
            _est_mean_tensor_descriptor = std::make_unique<Miopen_tensor>(*est_mean_attr->second);
        }
    }

    if(attributes.inv_variance().has_value())
    {
        if(auto inv_variance_attr = tensor_map.find(attributes.inv_variance().value());
           inv_variance_attr != tensor_map.end())
        {
            _est_variance_tensor_descriptor
                = std::make_unique<Miopen_tensor>(*inv_variance_attr->second);
        }
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
    float alpha = static_cast<float>(1);
    float beta = static_cast<float>(0);
    double epsilon = 1e-3; // taken from bn driver, todo, figure out better way

    auto x_buffer = miopen_utils::find_device_buffer(
        _inference_params->x().uid(), device_buffers, num_device_buffers);
    auto y_buffer = miopen_utils::find_device_buffer(
        _inference_params->y().uid(), device_buffers, num_device_buffers);
    auto scale_buffer = miopen_utils::find_device_buffer(
        _inference_params->scale().uid(), device_buffers, num_device_buffers);
    auto bias_buffer = miopen_utils::find_device_buffer(
        _inference_params->bias().uid(), device_buffers, num_device_buffers);

    hipdnnPluginDeviceBuffer_t est_mean_buffer = {0, nullptr};
    if(auto est_mean = _inference_params->est_mean(); est_mean.has_value())
    {
        est_mean_buffer = miopen_utils::find_device_buffer(
            est_mean.value()->uid(), device_buffers, num_device_buffers);
    }

    hipdnnPluginDeviceBuffer_t est_variance_buffer = {0, nullptr};
    if(auto est_variance = _inference_params->est_variance(); est_variance.has_value())
    {
        est_variance_buffer = miopen_utils::find_device_buffer(
            est_variance.value()->uid(), device_buffers, num_device_buffers);
    }

    auto miopen_status = miopenBatchNormalizationForwardInference_V2(
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
        epsilon);

    HIPDNN_LOG_INFO("MIOpen batchnorm forward inference status: {}",
                    miopenGetErrorString(miopen_status));
}

}