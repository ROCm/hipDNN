// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// NOLINTBEGIN(portability-template-virtual-member-function)

#include <cstdint>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/Tensor.hpp>
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

    virtual void batchnormFwdInference(const TensorBase<InputDataType>& input,
                                       const TensorBase<ScaleBiasDataType>& scale,
                                       const TensorBase<ScaleBiasDataType>& bias,
                                       const TensorBase<MeanVarianceDataType>& estimatedMean,
                                       const TensorBase<MeanVarianceDataType>& estimatedVariance,
                                       TensorBase<InputDataType>& output,
                                       double epsilon)
        = 0;

    // Could call this bwd_training or bwd_propagation
    virtual void batchnormBwd(const TensorBase<InputDataType>& dy,
                              const TensorBase<InputDataType>& x,
                              const TensorBase<MeanVarianceDataType>& mean,
                              const TensorBase<MeanVarianceDataType>& invVariance,
                              const TensorBase<ScaleBiasDataType>& scale,
                              TensorBase<InputDataType>& dx,
                              TensorBase<ScaleBiasDataType>& dscale,
                              TensorBase<ScaleBiasDataType>& dbias)
        = 0;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk

// NOLINTEND(portability-template-virtual-member-function)
