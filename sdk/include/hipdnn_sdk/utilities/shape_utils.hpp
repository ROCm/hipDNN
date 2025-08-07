// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

namespace hipdnn_sdk
{
namespace utilities
{

// Sets a default stride ordering based off the provided stride order.
// Ex. dim = {1,2,3,4} stride_order = {3, 0, 2, 1} for NHWC
// returns {24, 1, 8, 2}
inline std::vector<int64_t> generate_strides(const std::vector<int64_t>& dim,
                                             const std::vector<int64_t>& stride_order)
{
    size_t num_dims = dim.size();
    std::vector<int64_t> stride(num_dims, 1);

    // Create a mapping of stride order to dimension index
    std::vector<size_t> indices(num_dims);
    std::iota(indices.begin(), indices.end(), 0);
    std::ranges::sort(indices.begin(), indices.end(), [&stride_order](size_t a, size_t b) {
        return stride_order[a] < stride_order[b];
    });

    int64_t accumulator = 1;
    for(auto idx : indices)
    {
        stride[idx] = accumulator;
        accumulator *= dim[idx];
    }

    return stride;
}

// Sets stride order as NHWC for the provided dims.
// Ex. 4 will return {3, 0, 2, 1} for NHWC
inline std::vector<int64_t> stride_order_nhwc(size_t num_dims)
{
    // Default all to 0, and set everything up until NC
    std::vector<int64_t> stride_order(num_dims, 0);

    if(num_dims < 2)
    {
        return stride_order;
    }

    int64_t order = 1;
    for(size_t i = num_dims - 1; i > 1; --i)
    {
        stride_order[i] = order++;
    }
    stride_order[0] = order;

    return stride_order;
}

}
}
