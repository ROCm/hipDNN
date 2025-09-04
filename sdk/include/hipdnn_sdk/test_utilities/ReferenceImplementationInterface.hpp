// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <string>
#include <vector>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

template <template <class...> class BatchnormRefType, template <class...> class ConvRefType>
class BaseReferenceContainer
{

public:
    BaseReferenceContainer() = default;
    ~BaseReferenceContainer() = default;

    // === Batchnorm Operations ===

    template <typename InputDataType,
              typename ScaleBiasDataType,
              typename MeanVarianceDataType = ScaleBiasDataType>
    void batchnormFwdInference(const ITensor<InputDataType>& input,
                               const ITensor<ScaleBiasDataType>& scale,
                               const ITensor<ScaleBiasDataType>& bias,
                               const ITensor<MeanVarianceDataType>& estimatedMean,
                               const ITensor<MeanVarianceDataType>& estimatedVariance,
                               ITensor<InputDataType>& output,
                               double epsilon)
    {
        BatchnormRefType<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormFwdInference(
                input, scale, bias, estimatedMean, estimatedVariance, output, epsilon);
    }

    template <typename InputDataType,
              typename ScaleBiasDataType,
              typename MeanVarianceDataType = ScaleBiasDataType>
    void batchnormBwd(const ITensor<InputDataType>& dy,
                      const ITensor<InputDataType>& x,
                      const ITensor<MeanVarianceDataType>& mean,
                      const ITensor<MeanVarianceDataType>& invVariance,
                      const ITensor<ScaleBiasDataType>& scale,
                      ITensor<InputDataType>& dx,
                      ITensor<ScaleBiasDataType>& dscale,
                      ITensor<ScaleBiasDataType>& dbias)
    {
        BatchnormRefType<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::batchnormBwd(
            dy, x, mean, invVariance, scale, dx, dscale, dbias);
    }

    // === Convolution Operations ===

    template <typename InputDataType>
    void convFwdInference(const ITensor<InputDataType>& input,
                          const ITensor<InputDataType>& weight,
                          ITensor<InputDataType>& output,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations,
                          const std::vector<int64_t>& padding)
    {
        ConvRefType<InputDataType, InputDataType, InputDataType>::convFwdInference(
            input, weight, output, strides, dilations, padding);
    }
};

using CpuReferenceContainer
    = BaseReferenceContainer<CpuFpReferenceBatchnormImpl, CpuFpReferenceConvolutionImpl>;

// Future GPU implementation:
// using GpuReferenceContainer = BaseReferenceContainer<GpuFpReferenceBatchnormImpl, GpuFpReferenceConvolutionImpl>;

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
