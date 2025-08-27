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
inline std::vector<int64_t> generateStrides(const std::vector<int64_t>& dim,
                                            const std::vector<int64_t>& strideOrder)
{
    size_t numDims = dim.size();
    std::vector<int64_t> stride(numDims, 1);

    // Create a mapping of stride order to dimension index
    std::vector<size_t> indices(numDims);
    std::iota(indices.begin(), indices.end(), 0);
    std::ranges::sort(indices.begin(), indices.end(), [&strideOrder](size_t a, size_t b) {
        return strideOrder[a] < strideOrder[b];
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
inline std::vector<int64_t> strideOrderNhwc(size_t numDims)
{
    // Default all to 0, and set everything up until NC
    std::vector<int64_t> strideOrder(numDims, 0);

    if(numDims < 2)
    {
        return strideOrder;
    }

    int64_t order = 1;
    for(size_t i = numDims - 1; i > 1; --i)
    {
        strideOrder[i] = order++;
    }
    strideOrder[0] = order;

    return strideOrder;
}

}
}
