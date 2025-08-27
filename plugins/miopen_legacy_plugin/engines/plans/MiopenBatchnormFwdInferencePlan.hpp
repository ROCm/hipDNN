// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include "PlanBuilderInterface.hpp" //temporary
#include "PlanInterface.hpp"
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class BatchnormFwdInferenceParams
{
public:
    BatchnormFwdInferenceParams(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    const MiopenTensor& x() const;
    const MiopenTensor& y() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& bias() const;
    const std::optional<std::unique_ptr<MiopenTensor>>& estMean() const;
    const std::optional<std::unique_ptr<MiopenTensor>>& estVariance() const;

private:
    void initializeTensors(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    std::unique_ptr<MiopenTensor> _xPair;
    std::unique_ptr<MiopenTensor> _yPair;
    std::unique_ptr<MiopenTensor> _scalePair;
    std::unique_ptr<MiopenTensor> _biasPair;

    std::optional<std::unique_ptr<MiopenTensor>> _estMeanTensorDescriptor;
    std::optional<std::unique_ptr<MiopenTensor>> _estVarianceTensorDescriptor;
};

class BatchnormFwdInferencePlan : public IPlan
{
public:
    BatchnormFwdInferencePlan(std::unique_ptr<BatchnormFwdInferenceParams> inferenceParams);

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    std::unique_ptr<BatchnormFwdInferenceParams> _inferenceParams;
};

}
