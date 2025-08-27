// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormFwdInferencePlan.hpp"

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    initializeTensors(attributes, tensorMap);
}

const MiopenTensor& BatchnormFwdInferenceParams::x() const
{
    return *_xPair;
}

const MiopenTensor& BatchnormFwdInferenceParams::y() const
{
    return *_yPair;
}

const MiopenTensor& BatchnormFwdInferenceParams::scale() const
{
    return *_scalePair;
}

const MiopenTensor& BatchnormFwdInferenceParams::bias() const
{
    return *_biasPair;
}

const std::optional<std::unique_ptr<MiopenTensor>>& BatchnormFwdInferenceParams::estMean() const
{
    return _estMeanTensorDescriptor;
}

const std::optional<std::unique_ptr<MiopenTensor>>& BatchnormFwdInferenceParams::estVariance() const
{
    return _estVarianceTensorDescriptor;
}

namespace
{

std::unique_ptr<MiopenTensor> createTensor(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    int64_t uid)
{
    if(auto tensorAttr = tensorMap.find(uid); tensorAttr != tensorMap.end())
    {
        return std::make_unique<MiopenTensor>(*tensorAttr->second);
    }

    throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                               "Failed to find tensor with UID in tensorMap: "
                                                   + std::to_string(uid));
}

} // namespace

void BatchnormFwdInferenceParams::initializeTensors(
    const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
{
    _xPair = createTensor(tensorMap, attributes.x_tensor_uid());
    _yPair = createTensor(tensorMap, attributes.y_tensor_uid());
    _scalePair = createTensor(tensorMap, attributes.scale_tensor_uid());
    _biasPair = createTensor(tensorMap, attributes.bias_tensor_uid());

    if(attributes.mean_tensor_uid().has_value())
    {
        _estMeanTensorDescriptor = createTensor(tensorMap, attributes.mean_tensor_uid().value());
    }
    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _estVarianceTensorDescriptor
            = createTensor(tensorMap, attributes.inv_variance_tensor_uid().value());
    }
}

BatchnormFwdInferencePlan::BatchnormFwdInferencePlan(
    std::unique_ptr<BatchnormFwdInferenceParams> inferenceParams)
    : _inferenceParams(std::move(inferenceParams))
{
}

void BatchnormFwdInferencePlan::execute(const HipdnnEnginePluginHandle& handle,
                                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers,
                                        void* workspace) const
{
    std::ignore = workspace;

    // Hardcoded values from bn_driver in miopen
    auto alpha = static_cast<float>(1);
    auto beta = static_cast<float>(0);
    double epsilon = 1e-3;

    auto xBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams->x().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams->y().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams->scale().uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = miopen_utils::findDeviceBuffer(
        _inferenceParams->bias().uid(), deviceBuffers, numDeviceBuffers);

    hipdnnPluginDeviceBuffer_t estMeanBuffer = {0, nullptr};
    if(_inferenceParams->estMean().has_value())
    {
        estMeanBuffer = miopen_utils::findDeviceBuffer(
            _inferenceParams->estMean().value()->uid(), deviceBuffers, numDeviceBuffers);
    }

    hipdnnPluginDeviceBuffer_t estVarianceBuffer = {0, nullptr};
    if(_inferenceParams->estVariance().has_value())
    {
        estVarianceBuffer = miopen_utils::findDeviceBuffer(
            _inferenceParams->estVariance().value()->uid(), deviceBuffers, numDeviceBuffers);
    }

    THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationForwardInference_V2(
        handle.miopenHandle,
        MIOPEN_BATCHNORM_MODE,
        &alpha,
        &beta,
        _inferenceParams->x().tensorDescriptor(),
        xBuffer.ptr,
        _inferenceParams->y().tensorDescriptor(),
        yBuffer.ptr,
        _inferenceParams->scale().tensorDescriptor(),
        _inferenceParams->bias().tensorDescriptor(),
        _inferenceParams->estMean().has_value()
            ? _inferenceParams->estMean().value()->tensorDescriptor()
            : nullptr,
        _inferenceParams->estVariance().has_value()
            ? _inferenceParams->estVariance().value()->tensorDescriptor()
            : nullptr,
        scaleBuffer.ptr,
        biasBuffer.ptr,
        estMeanBuffer.ptr,
        estVarianceBuffer.ptr,
        epsilon));
}

}
