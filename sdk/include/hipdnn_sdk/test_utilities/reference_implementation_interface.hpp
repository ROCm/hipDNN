// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

#include <cstdint>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/tensor.hpp>
#include <map>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

template <class InputDataType,
          class ScaleBiasDataType,
          class MeanVarianceDataType = ScaleBiasDataType>
class IReferenceImplementation
{
public:
    virtual ~IReferenceImplementation() = default;

    virtual void
        batchnormFwdInference(const Tensor_interface<InputDataType>& input,
                              const Tensor_interface<ScaleBiasDataType>& scale,
                              const Tensor_interface<ScaleBiasDataType>& bias,
                              const Tensor_interface<MeanVarianceDataType>& estimatedMean,
                              const Tensor_interface<MeanVarianceDataType>& estimatedVariance,
                              Tensor_interface<InputDataType>& output,
                              double epsilon)
        = 0;

    // Could call this bwd_training or bwd_propagation
    virtual void batchnormBwd(const Tensor_interface<InputDataType>& dy,
                              const Tensor_interface<InputDataType>& x,
                              const Tensor_interface<MeanVarianceDataType>& mean,
                              const Tensor_interface<MeanVarianceDataType>& invVariance,
                              const Tensor_interface<ScaleBiasDataType>& scale,
                              Tensor_interface<InputDataType>& dx,
                              Tensor_interface<ScaleBiasDataType>& dscale,
                              Tensor_interface<ScaleBiasDataType>& dbias)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)
