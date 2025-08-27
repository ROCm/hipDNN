// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes/tensor_attributes.hpp"
#include "error.hpp"
#include <algorithm>
#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>
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
inline error_t findCommonShape(const std::vector<std::vector<int64_t>>& inputShapes,
                               std::vector<int64_t>& commonShape)
{
    if(inputShapes.empty())
    {
        return {error_code_t::INVALID_VALUE, "Input shapes cannot be empty"};
    }

    size_t dims = std::ranges::max_element(
                      inputShapes.begin(),
                      inputShapes.end(),
                      [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
                          return a.size() < b.size();
                      })
                      ->size();

    commonShape.resize(dims, 1);

    for(auto& current : inputShapes)
    {
        for(size_t j = current.size(); j-- > 0;)
        {
            if(commonShape[j] != current[j] && commonShape[j] != 1 && current[j] != 1)
            {
                return {error_code_t::INVALID_VALUE, "Incompatible shapes"};
            }

            commonShape[j] = std::max(commonShape[j], current[j]);
        }
    }

    return {};
}

// Utility function to create Tensor_attributes from a Tensor
template <class T,
          class HostAlloc = hipdnn_sdk::utilities::Host_allocator<T>,
          class DeviceAlloc = hipdnn_sdk::utilities::Device_allocator<T>>
inline TensorAttributes
    makeTensorAttributes(const std::string& name,
                         DataType_t dataType,
                         const hipdnn_sdk::utilities::Tensor<T, HostAlloc, DeviceAlloc>& tensor)
{
    return TensorAttributes()
        .set_name(name)
        .set_data_type(dataType)
        .set_dim(tensor.dims())
        .set_stride(tensor.strides());
}

}

inline int32_t initializeFrontendLogging(hipdnnCallback_t fn)
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
