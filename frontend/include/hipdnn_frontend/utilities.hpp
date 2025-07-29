// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "error.hpp"
#include <algorithm>
#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/logger.hpp>
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

inline int32_t initialize_frontend_logging(hipdnnCallback_t fn)
{
    if(fn == nullptr)
    {
        return -1;
    }

#ifdef COMPONENT_NAME
    hipdnn::logging::initialize_callback_logging(COMPONENT_NAME, fn);
#else
    return -1;
#endif

    HIPDNN_LOG_INFO("Frontend logging initialized via callback.");

    return 0;
}
}
}