// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "miopen_batchnorm_bwd_plan.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

Batchnorm_bwd_params::Batchnorm_bwd_params(
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensor_map)
{
    initialize_tensors(attributes, tensor_map);
}

const Miopen_tensor& Batchnorm_bwd_params::x() const
{
    return *_x;
}

const Miopen_tensor& Batchnorm_bwd_params::dy() const
{
    return *_dy;
}

const Miopen_tensor& Batchnorm_bwd_params::dx() const
{
    return *_dx;
}

const Miopen_tensor& Batchnorm_bwd_params::scale() const
{
    return *_scale;
}

const Miopen_tensor& Batchnorm_bwd_params::dscale() const
{
    return *_dscale;
}

const Miopen_tensor& Batchnorm_bwd_params::dbias() const
{
    return *_dbias;
}

const std::optional<std::unique_ptr<Miopen_tensor>>& Batchnorm_bwd_params::opt_mean() const
{
    return _opt_mean;
}

const std::optional<std::unique_ptr<Miopen_tensor>>& Batchnorm_bwd_params::opt_inv_variance() const
{
    return _opt_inv_variance;
}

namespace
{

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

} // namespace

void Batchnorm_bwd_params::initialize_tensors(
    const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
        tensor_map)
{
    _x = create_tensor(tensor_map, attributes.x_tensor_uid());
    _dy = create_tensor(tensor_map, attributes.dy_tensor_uid());
    _dx = create_tensor(tensor_map, attributes.dx_tensor_uid());
    _scale = create_tensor(tensor_map, attributes.scale_tensor_uid());
    _dscale = create_tensor(tensor_map, attributes.dscale_tensor_uid());
    _dbias = create_tensor(tensor_map, attributes.dbias_tensor_uid());

    if(attributes.mean_tensor_uid().has_value())
    {
        _opt_mean = create_tensor(tensor_map, attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _opt_inv_variance = create_tensor(tensor_map, attributes.inv_variance_tensor_uid().value());
    }
}

Batchnorm_bwd_plan::Batchnorm_bwd_plan(std::unique_ptr<Batchnorm_bwd_params> params)
    : _params(std::move(params))
{
}

void Batchnorm_bwd_plan::execute(const hipdnnEnginePluginHandle& handle,
                                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                                 uint32_t num_device_buffers,
                                 void* workspace) const
{
    std::ignore = workspace;

    float alpha_data_diff = 1.0f;
    float beta_data_diff = 0.0f;
    float alpha_param_diff = 1.0f;
    float beta_param_diff = 0.0f;
    double epsilon = 1e-3;

    auto x_buffer
        = miopen_utils::find_device_buffer(_params->x().uid(), device_buffers, num_device_buffers);
    auto dy_buffer
        = miopen_utils::find_device_buffer(_params->dy().uid(), device_buffers, num_device_buffers);
    auto dx_buffer
        = miopen_utils::find_device_buffer(_params->dx().uid(), device_buffers, num_device_buffers);
    auto scale_buffer = miopen_utils::find_device_buffer(
        _params->scale().uid(), device_buffers, num_device_buffers);
    auto dscale_buffer = miopen_utils::find_device_buffer(
        _params->dscale().uid(), device_buffers, num_device_buffers);
    auto dbias_buffer = miopen_utils::find_device_buffer(
        _params->dbias().uid(), device_buffers, num_device_buffers);

    hipdnnPluginDeviceBuffer_t mean_buffer = {0, nullptr};
    if(_params->opt_mean().has_value())
    {
        mean_buffer = miopen_utils::find_device_buffer(
            _params->opt_mean().value()->uid(), device_buffers, num_device_buffers);
    }

    hipdnnPluginDeviceBuffer_t inv_variance_buffer = {0, nullptr};
    if(_params->opt_inv_variance().has_value())
    {
        inv_variance_buffer = miopen_utils::find_device_buffer(
            _params->opt_inv_variance().value()->uid(), device_buffers, num_device_buffers);
    }

    THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationBackward_V2(
        handle.miopen_handle,
        MIOPEN_BATCHNORM_MODE,
        &alpha_data_diff,
        &beta_data_diff,
        &alpha_param_diff,
        &beta_param_diff,
        _params->x().tensor_descriptor(),
        x_buffer.ptr,
        _params->dy().tensor_descriptor(),
        dy_buffer.ptr,
        _params->dx().tensor_descriptor(),
        dx_buffer.ptr,
        _params->scale().tensor_descriptor(),
        _params->scale().tensor_descriptor(),
        _params->opt_mean().has_value() ? _params->opt_mean().value()->tensor_descriptor()
                                        : nullptr,
        _params->opt_inv_variance().has_value()
            ? _params->opt_inv_variance().value()->tensor_descriptor()
            : nullptr,
        scale_buffer.ptr,
        dscale_buffer.ptr,
        dbias_buffer.ptr,
        epsilon,
        mean_buffer.ptr,
        inv_variance_buffer.ptr));
}

} // namespace miopen_legacy_plugin
