// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/ReferenceImplementationInterface.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>

#if defined(__HIP_PLATFORM_AMD__)
// Need these for the half and bfloat16 types
#include <hipdnn_sdk/utilities/HalfUtils.hpp>
#include <hipdnn_sdk/utilities/HipBfloat16Utils.hpp>
#endif

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class InputDataType,
          class ScaleBiasDataType,
          class MeanVarianceDataType = ScaleBiasDataType>
class CpuFpReferenceImplementation
    : public IReferenceImplementation<InputDataType, ScaleBiasDataType, MeanVarianceDataType>
{
public:
    CpuFpReferenceImplementation() = default;
    ~CpuFpReferenceImplementation() override = default;

    void batchnormFwdInference(const ITensor<InputDataType>& input,
                               const ITensor<ScaleBiasDataType>& scale,
                               const ITensor<ScaleBiasDataType>& bias,
                               const ITensor<MeanVarianceDataType>& estimatedMean,
                               const ITensor<MeanVarianceDataType>& estimatedVariance,
                               ITensor<InputDataType>& output,
                               double epsilon) override
    {
        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormFwdInference(input, scale, bias, estimatedMean, estimatedVariance, output, epsilon);
    }

    void batchnormBwd(const ITensor<InputDataType>& dy,
                      const ITensor<InputDataType>& x,
                      const ITensor<MeanVarianceDataType>& mean,
                      const ITensor<MeanVarianceDataType>& invVariance,
                      const ITensor<ScaleBiasDataType>& scale,
                      ITensor<InputDataType>& dx,
                      ITensor<ScaleBiasDataType>& dscale,
                      ITensor<ScaleBiasDataType>& dbias) override
    {
        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormBwd(dy, x, mean, invVariance, scale, dx, dscale, dbias);
    }

    void convFwdInference(const ITensor<InputDataType>& input,
                          const ITensor<InputDataType>& weight,
                          ITensor<InputDataType>& output,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations,
                          const std::vector<int64_t>& padding) override
    {
        CpuFpReferenceConvolutionImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            convFwdInference(input, weight, output, strides, dilations, padding);
    }
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
