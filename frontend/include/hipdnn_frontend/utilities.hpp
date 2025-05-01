// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "error.hpp"
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
// Find common shape from inputs.
// Takes the max in each dim, and if any dim is not 1, or equal, then it's incompatible.
// For example:
// input_shapes = {{1, 2}, {1, 2}, {1, 2, 5}} -> common_shape = {1, 2, 5}
// input_shapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}} -> error
inline error_t find_common_shape(const std::vector<std::vector<int64_t>>& input_shapes,
                                 std::vector<int64_t>& common_shape)
{
    if(input_shapes.empty())
    {
        return {error_code_t::INVALID_VALUE, "Input shapes cannot be empty"};
    }

    size_t dims = std::ranges::max_element(
                      input_shapes.begin(),
                      input_shapes.end(),
                      [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
                          return a.size() < b.size();
                      })
                      ->size();

    common_shape.resize(dims, 1);

    for(auto& current : input_shapes)
    {
        for(size_t j = current.size(); j-- > 0;)
        {
            if(common_shape[j] != current[j] && common_shape[j] != 1 && current[j] != 1)
            {
                return {error_code_t::INVALID_VALUE, "Incompatible shapes"};
            }

            common_shape[j] = std::max(common_shape[j], current[j]);
        }
    }

    return {};
}

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