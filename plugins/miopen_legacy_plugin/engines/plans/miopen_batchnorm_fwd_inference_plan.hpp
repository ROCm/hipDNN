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

class BatchnormFwdInferenceParams
{
public:
    BatchnormFwdInferenceParams(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    const Miopen_tensor& x() const;
    const Miopen_tensor& y() const;
    const Miopen_tensor& scale() const;
    const Miopen_tensor& bias() const;
    const std::optional<std::unique_ptr<Miopen_tensor>>& estMean() const;
    const std::optional<std::unique_ptr<Miopen_tensor>>& estVariance() const;

private:
    void initializeTensors(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    std::unique_ptr<Miopen_tensor> _xPair;
    std::unique_ptr<Miopen_tensor> _yPair;
    std::unique_ptr<Miopen_tensor> _scalePair;
    std::unique_ptr<Miopen_tensor> _biasPair;

    std::optional<std::unique_ptr<Miopen_tensor>> _estMeanTensorDescriptor;
    std::optional<std::unique_ptr<Miopen_tensor>> _estVarianceTensorDescriptor;
};

class BatchnormFwdInferencePlan : public PlanInterface
{
public:
    BatchnormFwdInferencePlan(std::unique_ptr<BatchnormFwdInferenceParams> inferenceParams);

    void execute(const hipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<BatchnormFwdInferenceParams> _inferenceParams;
};

}
