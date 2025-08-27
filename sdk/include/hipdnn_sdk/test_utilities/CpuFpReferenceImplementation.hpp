// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/ReferenceImplementationInterface.hpp>
#include <numeric>
#include <vector>

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
            auto mean = estimatedMean.getHostValue(0, cidx, 0, 0);
            auto variance = estimatedVariance.getHostValue(0, cidx, 0, 0);
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
                            input.getHostValue(bidx, cidx, row, column));
                        MeanVarianceDataType elemStd = in - mean;
                        MeanVarianceDataType inhat = elemStd * invVariance;
                        output.setHostValue(
                            bidx,
                            cidx,
                            row,
                            column,
                            static_cast<InputDataType>((scale.getHostValue(0, cidx, 0, 0)
                                                        * static_cast<ScaleBiasDataType>(inhat))
                                                       + bias.getHostValue(0, cidx, 0, 0)));
                    }
                }
            }
        });

        output.memory().markHostModified(); // Mark output memory as modified on host
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
            auto channelMean = mean.getHostValue(0, cidx, 0, 0);
            auto channelInvVariance
                = invVariance.getHostValue(0, cidx, 0, 0); // 1 / sqrt(var + eps)
            auto channelScale = scale.getHostValue(0, cidx, 0, 0);

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
                            x.getHostValue(bidx, cidx, row, column));
                        auto dyVal = static_cast<MeanVarianceDataType>(
                            dy.getHostValue(bidx, cidx, row, column));

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

            dscale.setHostValue(0, cidx, 0, 0, static_cast<ScaleBiasDataType>(dotProduct));

            dbias.setHostValue(0, cidx, 0, 0, static_cast<ScaleBiasDataType>(sumDy));

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
                            x.getHostValue(bidx, cidx, row, column));
                        auto dyVal = static_cast<MeanVarianceDataType>(
                            dy.getHostValue(bidx, cidx, row, column));

                        MeanVarianceDataType xHat = (xVal - channelMean) * channelInvVariance;
                        MeanVarianceDataType dxVal
                            = (dyVal - meanDy - xHat * meanDyXhat) * scalarCoef;

                        dx.setHostValue(bidx, cidx, row, column, static_cast<InputDataType>(dxVal));
                    }
                }
            }
        });

        dx.memory().markHostModified();
        dscale.memory().markHostModified();
        dbias.memory().markHostModified();
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
