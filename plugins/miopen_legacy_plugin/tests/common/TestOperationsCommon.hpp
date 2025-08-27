// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <iostream>
#include <vector>

namespace test_operations_common
{

using namespace hipdnn_sdk::reference_test_utilities;

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
    generateRandomDeviceBuffer(ITensor<T>& tensor, int uid, T min, T max, unsigned int seed = 0)
{
    tensor.fillWithRandomValues(min, max, seed);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateStaticDeviceBuffer(ITensor<T>& tensor, int uid, T value)
{
    tensor.fillWithValue(value);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generateEmptyDeviceBuffer(ITensor<T>& tensor, int uid)
{
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().deviceData();
    return buffer;
}

inline std::vector<Batchnorm2dTestCase> getBatchnorm2dTestCases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 2, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 3, .h = 14, .w = 14},
    };
}

} // namespace test_operation_common
