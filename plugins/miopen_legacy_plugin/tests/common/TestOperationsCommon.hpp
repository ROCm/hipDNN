// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/utilities/StringUtil.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <iostream>
#include <vector>

namespace test_operations_common
{

using namespace hipdnn_sdk::test_utilities;

struct ConvTestCase
{
    std::vector<int64_t> _xDims;
    std::vector<int64_t> _wDims;
    std::vector<int64_t> _yDims;
    std::vector<int64_t> _convPrePadding;
    std::vector<int64_t> _convPostPadding;
    std::vector<int64_t> _convStride;
    std::vector<int64_t> _convDilation;

    ConvTestCase(std::vector<int64_t>&& xDims,
                 std::vector<int64_t>&& wDims,
                 std::vector<int64_t>&& convPrePadding,
                 std::vector<int64_t>&& convPostPadding,
                 std::vector<int64_t>&& convStride,
                 std::vector<int64_t>&& convDilation)
        : _xDims(std::move(xDims))
        , _wDims(std::move(wDims))
        , _convPrePadding(std::move(convPrePadding))
        , _convPostPadding(std::move(convPostPadding))
        , _convStride(std::move(convStride))
        , _convDilation(std::move(convDilation))
    {
        // Indices for dimensions
        // N - Batch size, always at index 0
        // C - Channels, always at index 1
        // D - Depth (for 5D tensors), always at index 2 if present
        // H - Height, always at index 2 for 4D tensors and index 3 for 5D tensors
        // W - Width, always at index 3 for 4D tensors and index 4 for 5D tensors
        constexpr int N = 0; // Batch size index

        if(_xDims.size() != _wDims.size())
        {
            throw std::invalid_argument("xDims and wDims must have the same number of dimensions.");
        }

        // Ensure xDims has at least 3 dimensions (N, C, and at least 1 spatial dimension)
        if(_xDims.size() < 3)
        {
            throw std::invalid_argument(
                "xDims must have at least 3 dimensions (N, C, and at least 1 spatial dimension).");
        }

        // Determine the number of spatial dimensions
        auto spatialDims = _xDims.size() - 2; // Exclude N and C

        // Validate that the convolution parameter vectors match the number of spatial dimensions
        if(_convPrePadding.size() != spatialDims || _convPostPadding.size() != spatialDims
           || _convDilation.size() != spatialDims || _convStride.size() != spatialDims)
        {
            throw std::invalid_argument(
                "Convolution parameter vectors must match the number of spatial dimensions.");
        }

        // Calculate output dimensions based on input dimensions and convolution parameters
        auto n = _xDims[N];
        auto cOut = _wDims[N];
        std::vector<int64_t> outputDims = {n, cOut};

        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto paddedInputSize = _xDims[2 + i] + _convPrePadding[i] + _convPostPadding[i];
            auto effectiveKernelSize = _convDilation[i] * (_wDims[2 + i] - 1) + 1;
            auto dimOut = ((paddedInputSize - effectiveKernelSize) / _convStride[i]) + 1;
            outputDims.push_back(dimOut);
        }

        _yDims = outputDims;
    }

    friend std::ostream& operator<<(std::ostream& ss, const ConvTestCase& tc)
    {
        ss << "(x:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._xDims);
        ss << " w:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._wDims);
        ss << " y:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._yDims);
        ss << " prePad:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._convPrePadding);
        ss << " postPad:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._convPostPadding);
        ss << " stride:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._convStride);
        ss << " dilation:";
        hipdnn_sdk::test_utilities::vecToStream(ss, tc._convDilation);
        ss << ")";

        return ss;
    }
};

struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

template <typename T>
hipdnnPluginDeviceBuffer_t
    generateRandomDeviceBuffer(TensorBase<T>& tensor, int uid, T min, T max, unsigned int seed = 0)
{
    tensor.fillWithRandomValues(min, max, seed);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateStaticDeviceBuffer(TensorBase<T>& tensor, int uid, T value)
{
    tensor.fillWithValue(value);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateEmptyDeviceBuffer(TensorBase<T>& tensor, int uid)
{
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

inline std::vector<ConvTestCase> getConvTestCases()
{
    return {
        {{1, 20, 20, 20}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
        {{1, 20, 20, 20}, {1, 1, 3, 3}, {0, 0}, {0, 0}, {1, 1}, {1, 1}},
        {{1, 20, 20, 20}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1}},
        {{1, 20, 20, 20}, {1, 1, 3, 3}, {2, 2}, {2, 2}, {1, 1}, {2, 2}},
    };
}

inline std::vector<Batchnorm2dTestCase> getBatchnorm2dTestCases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 2, .c = 3, .h = 14, .w = 14},
    };
}

} // namespace test_operation_common
