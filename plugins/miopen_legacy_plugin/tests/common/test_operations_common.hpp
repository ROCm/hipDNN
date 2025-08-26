// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/utilities/tensor.hpp>
#include <iostream>
#include <vector>

namespace test_operations_common
{

using namespace hipdnn_sdk::reference_test_utilities;

struct Bn_2d_test_case
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;

    friend std::ostream& operator<<(std::ostream& ss, const Bn_2d_test_case& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w << ")";
    }

    std::vector<int64_t> get_dims() const
    {
        return {n, c, h, w};
    }
};

template <typename T>
hipdnnPluginDeviceBuffer_t generate_random_device_buffer(
    Tensor_interface<T>& tensor, int uid, T min, T max, unsigned int seed = 0)
{
    tensor.fill_with_random_values(min, max, seed);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().device_data();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t
    generate_static_device_buffer(Tensor_interface<T>& tensor, int uid, T value)
{
    tensor.fill_with_value(value);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().device_data();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generate_empty_device_buffer(Tensor_interface<T>& tensor, int uid)
{
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().device_data();
    return buffer;
}

inline std::vector<Bn_2d_test_case> get_bn_2d_test_cases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 2, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 3, .h = 14, .w = 14},
    };
}

} // namespace test_operation_common
