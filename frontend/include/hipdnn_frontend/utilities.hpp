// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "error.hpp"
#include <algorithm>
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
                                 std::vector<int64_t>&                   common_shape)
{
    if(input_shapes.empty())
    {
        return {error_code_t::INVALID_VALUE, "Input shapes cannot be empty"};
    }

    int64_t dims
        = std::max_element(input_shapes.begin(),
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
}
}