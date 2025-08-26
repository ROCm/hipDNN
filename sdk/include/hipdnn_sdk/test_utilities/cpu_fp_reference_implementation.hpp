// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/reference_implementation_interface.hpp>
#include <numeric>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
// Need these for the half and bfloat16 types
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>
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

    void batchnormFwdInference(const Tensor_interface<InputDataType>& input,
                               const Tensor_interface<ScaleBiasDataType>& scale,
                               const Tensor_interface<ScaleBiasDataType>& bias,
                               const Tensor_interface<MeanVarianceDataType>& estimatedMean,
                               const Tensor_interface<MeanVarianceDataType>& estimatedVariance,
                               Tensor_interface<InputDataType>& output,
                               double epsilon) override
    {
        if(input.dims().size() != 4)
        {
            throw std::runtime_error("Batchnorm inference requires a 4D tensor.");
        }

        int64_t nBatches = input.dims().at(0);
        std::vector<int64_t> channels(static_cast<size_t>(input.dims().at(1)));
        std::iota(channels.begin(), channels.end(), 0);
        int64_t height = input.dims().at(2);
        int64_t width = input.dims().at(3);

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            auto mean = estimatedMean.get_host_value(0, cidx, 0, 0);
            auto variance = estimatedVariance.get_host_value(0, cidx, 0, 0);
            MeanVarianceDataType invVariance
                = static_cast<MeanVarianceDataType>(1.0f)
                  / sqrtInternal(variance + static_cast<MeanVarianceDataType>(epsilon));

            // process the batch per channel
            for(int bidx = 0; bidx < nBatches; bidx++)
            {
                for(int row = 0; row < height; row++)
                {
                    for(int column = 0; column < width; column++)
                    {
                        auto in = static_cast<MeanVarianceDataType>(
                            input.get_host_value(bidx, cidx, row, column));
                        MeanVarianceDataType elemStd = in - mean;
                        MeanVarianceDataType inhat = elemStd * invVariance;
                        output.set_host_value(
                            bidx,
                            cidx,
                            row,
                            column,
                            static_cast<InputDataType>((scale.get_host_value(0, cidx, 0, 0)
                                                        * static_cast<ScaleBiasDataType>(inhat))
                                                       + bias.get_host_value(0, cidx, 0, 0)));
                    }
                }
            }
        });

        output.memory().mark_host_modified(); // Mark output memory as modified on host
    }

    void batchnormBwd(const Tensor_interface<InputDataType>& dy,
                      const Tensor_interface<InputDataType>& x,
                      const Tensor_interface<MeanVarianceDataType>& mean,
                      const Tensor_interface<MeanVarianceDataType>& invVariance,
                      const Tensor_interface<ScaleBiasDataType>& scale,
                      Tensor_interface<InputDataType>& dx,
                      Tensor_interface<ScaleBiasDataType>& dscale,
                      Tensor_interface<ScaleBiasDataType>& dbias) override
    {
        if(x.dims().size() != 4)
        {
            throw std::runtime_error("Batchnorm backward requires a 4D tensor.");
        }

        int64_t nBatches = x.dims().at(0);
        int64_t nChannels = x.dims().at(1);
        int64_t height = x.dims().at(2);
        int64_t width = x.dims().at(3);
        int64_t nhw = nBatches * height * width; // Total elements per channel
        auto nhwF = static_cast<MeanVarianceDataType>(nhw);

        std::vector<int64_t> channels(static_cast<size_t>(nChannels));
        std::iota(channels.begin(), channels.end(), 0);

        std::for_each(channels.begin(), channels.end(), [&](int64_t cidx) {
            auto channelMean = mean.get_host_value(0, cidx, 0, 0);
            auto channelInvVariance
                = invVariance.get_host_value(0, cidx, 0, 0); // 1 / sqrt(var + eps)
            auto channelScale = scale.get_host_value(0, cidx, 0, 0);

            // Calculate dot product for (x - mean) * channelInvVariance * dy and ∑ dy for this channel
            MeanVarianceDataType dotProduct = 0;
            MeanVarianceDataType sumDy = 0;

            for(int bidx = 0; bidx < nBatches; bidx++)
            {
                for(int row = 0; row < height; row++)
                {
                    for(int column = 0; column < width; column++)
                    {
                        auto xVal = static_cast<MeanVarianceDataType>(
                            x.get_host_value(bidx, cidx, row, column));
                        auto dyVal = static_cast<MeanVarianceDataType>(
                            dy.get_host_value(bidx, cidx, row, column));

                        MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
                        dotProduct += xHat * dyVal;
                        sumDy += dyVal;
                    }
                }
            }

            // Per channel:
            // - dscale = ∑ (xHat * dy)
            // - dbias = ∑ dy
            // - dx = scale * invVariance * (dy - mean(dy) - xHat * mean(dy * xHat))

            dscale.set_host_value(0, cidx, 0, 0, static_cast<ScaleBiasDataType>(dotProduct));

            dbias.set_host_value(0, cidx, 0, 0, static_cast<ScaleBiasDataType>(sumDy));

            MeanVarianceDataType meanDy = sumDy / nhwF;
            MeanVarianceDataType meanDyXhat = dotProduct / nhwF;
            MeanVarianceDataType scalarCoef
                = static_cast<MeanVarianceDataType>(channelScale) * channelInvVariance;

            for(int bidx = 0; bidx < nBatches; bidx++)
            {
                for(int row = 0; row < height; row++)
                {
                    for(int column = 0; column < width; column++)
                    {
                        auto xVal = static_cast<MeanVarianceDataType>(
                            x.get_host_value(bidx, cidx, row, column));
                        auto dyVal = static_cast<MeanVarianceDataType>(
                            dy.get_host_value(bidx, cidx, row, column));

                        MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
                        MeanVarianceDataType dxVal
                            = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                        dx.set_host_value(
                            bidx, cidx, row, column, static_cast<InputDataType>(dxVal));
                    }
                }
            }
        });

        dx.memory().mark_host_modified();
        dscale.memory().mark_host_modified();
        dbias.memory().mark_host_modified();
    }

private:
    double sqrtInternal(double value) const
    {
        return std::sqrt(value);
    }

    float sqrtInternal(float value) const
    {
        return std::sqrtf(value);
    }
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
