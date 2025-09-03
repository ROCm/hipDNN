// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_sdk
{
namespace reference_test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class InputDataType,
          class ScaleBiasDataType,
          class MeanVarianceDataType = ScaleBiasDataType>
class CpuFpReferenceConvolutionImpl
{
public:
    static void convFwdInference(const ITensor<InputDataType>& input,
                                 const ITensor<InputDataType>& weight,
                                 ITensor<InputDataType>& output,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dilations,
                                 const std::vector<int64_t>& padding)
    {
        validateConvolutionInputs(input, weight, output, strides, dilations, padding);

        // Extract dimensions - NCHW format for input/output, [G*K][C][Y][X] for weight (4D flattened)
        const auto& inputDims = input.dims();
        const auto& weightDims = weight.dims();
        const auto& outputDims = output.dims();

        int64_t nBatch = inputDims[0];
        int64_t nInputChannels = inputDims[1];
        int64_t inputHeight = inputDims[2];
        int64_t inputWidth = inputDims[3];

        int64_t totalOutputChannels = weightDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = weightDims[1]; // C
        int64_t kernelHeight = weightDims[2]; // Y
        int64_t kernelWidth = weightDims[3]; // X

        int64_t outputHeight = outputDims[2];
        int64_t outputWidth = outputDims[3];

        // Calculate groups from input/weight channel relationship
        int64_t nGroups = nInputChannels / channelsPerGroup;
        int64_t outputChannelsPerGroup = totalOutputChannels / nGroups;

        // Extract convolution parameters
        int64_t strideH = strides[0];
        int64_t strideW = strides[1];
        int64_t dilationH = dilations[0];
        int64_t dilationW = dilations[1];
        int64_t padH = padding[0];
        int64_t padW = padding[1];

        // Convolution lambda for parallel execution
        auto convolutionFunc = [&](auto g, auto n, auto k, auto ho, auto wo) {
            float accumulator = 0.0f;

            // Convert template parameters to int64_t to avoid sign conversion warnings
            int64_t gIdx = static_cast<int64_t>(g);
            int64_t nIdx = static_cast<int64_t>(n);
            int64_t kIdx = static_cast<int64_t>(k);
            int64_t hoIdx = static_cast<int64_t>(ho);
            int64_t woIdx = static_cast<int64_t>(wo);

            // Input channels for this group
            int64_t baseInputChannel = gIdx * channelsPerGroup;

            // Perform convolution over input channels and kernel spatial dimensions
            for(int64_t c = 0; c < channelsPerGroup; ++c)
            {
                int64_t inputChannel = baseInputChannel + c;

                for(int64_t y = 0; y < kernelHeight; ++y)
                {
                    // Calculate input height coordinate
                    int64_t hi = hoIdx * strideH + y * dilationH - padH;

                    for(int64_t x = 0; x < kernelWidth; ++x)
                    {
                        // Calculate input width coordinate
                        int64_t wi = woIdx * strideW + x * dilationW - padW;

                        // Check bounds for valid input coordinates
                        if(hi >= 0 && hi < inputHeight && wi >= 0 && wi < inputWidth)
                        {
                            // Get input value
                            InputDataType inputVal = input.getHostValue(nIdx, inputChannel, hi, wi);

                            // Calculate weight index for 4D access: [G*K][C][Y][X] format
                            // Weight tensor first dimension is flattened: group*outputChannelsPerGroup + kernelIdx
                            int64_t weightIdx = gIdx * outputChannelsPerGroup + kIdx;
                            InputDataType weightVal = weight.getHostValue(weightIdx, c, y, x);

                            // Perform multiply-accumulate operation
                            accumulator
                                += static_cast<float>(inputVal) * static_cast<float>(weightVal);
                        }
                    }
                }
            }

            // Store result in output tensor (NCHW format: batch, channel, height, width)
            int64_t outputChannel = gIdx * outputChannelsPerGroup + kIdx;
            output.setHostValue(
                nIdx, outputChannel, hoIdx, woIdx, static_cast<InputDataType>(accumulator));
        };

        // Execute convolution in parallel across batch, groups, output channels, and spatial dimensions
        makeParallelTensorFunctor(
            convolutionFunc, nGroups, nBatch, outputChannelsPerGroup, outputHeight, outputWidth)(
            std::thread::hardware_concurrency());

        // Mark output as modified on host
        output.memory().markHostModified();
    }

private:
    static void validateConvolutionInputs(const ITensor<InputDataType>& input,
                                          const ITensor<InputDataType>& weight,
                                          const ITensor<InputDataType>& output,
                                          const std::vector<int64_t>& strides,
                                          const std::vector<int64_t>& dilations,
                                          const std::vector<int64_t>& padding)
    {
        // Validate tensor dimensions
        if(input.dims().size() != 4)
        {
            throw std::invalid_argument("Input tensor must be 4D (NCHW format)");
        }

        if(weight.dims().size() != 4)
        {
            throw std::invalid_argument("Weight tensor must be 4D ([G*K][C][Y][X] format)");
        }

        if(output.dims().size() != 4)
        {
            throw std::invalid_argument("Output tensor must be 4D (NCHW format)");
        }

        // Validate parameter vector sizes
        if(strides.size() != 2)
        {
            throw std::invalid_argument("Strides must have exactly 2 elements [H, W]");
        }

        if(dilations.size() != 2)
        {
            throw std::invalid_argument("Dilations must have exactly 2 elements [H, W]");
        }

        if(padding.size() != 2)
        {
            throw std::invalid_argument("Padding must have exactly 2 elements [H, W]");
        }

        // Validate parameter values
        for(auto stride : strides)
        {
            if(stride <= 0)
            {
                throw std::invalid_argument("All stride values must be positive");
            }
        }

        for(auto dilation : dilations)
        {
            if(dilation <= 0)
            {
                throw std::invalid_argument("All dilation values must be positive");
            }
        }

        for(auto pad : padding)
        {
            if(pad < 0)
            {
                throw std::invalid_argument("All padding values must be non-negative");
            }
        }

        // Validate tensor dimension compatibility (4D weight format: [G*K][C][Y][X])
        const auto& inputDims = input.dims();
        const auto& weightDims = weight.dims();
        const auto& outputDims = output.dims();

        int64_t inputChannels = inputDims[1];
        int64_t totalOutputChannels = weightDims[0]; // G * K
        int64_t channelsPerGroup = weightDims[1]; // C

        // Calculate number of groups
        if(inputChannels % channelsPerGroup != 0)
        {
            throw std::invalid_argument("Input channels must be divisible by channels per group");
        }

        int64_t nGroups = inputChannels / channelsPerGroup;

        if(totalOutputChannels % nGroups != 0)
        {
            throw std::invalid_argument(
                "Total output channels must be divisible by number of groups");
        }

        // Validate output dimensions
        if(outputDims[1] != totalOutputChannels)
        {
            throw std::invalid_argument("Output channel count mismatch");
        }

        if(inputDims[0] != outputDims[0])
        {
            throw std::invalid_argument("Batch size mismatch between input and output");
        }
    }
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk
