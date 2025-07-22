// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_sdk {
namespace reference_test_utilities {

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

// Wraps a TensorAttributes and Migratable_memory<T> to provide a common interface for testing
class Test_tensor
{
public:
    Test_tensor(const TensorAttributes& tensor_attributes, size_t item_size)
        : _memory(0, item_size)
    {
        // Convert FlatBuffers vectors to std::vectors
        if (tensor_attributes.dims()) {
            _dims.assign(tensor_attributes.dims()->begin(), tensor_attributes.dims()->end());
        }
        
        if (tensor_attributes.strides()) {
            _strides.assign(tensor_attributes.strides()->begin(), tensor_attributes.strides()->end());
        }

        if (_dims.empty() || _strides.empty()) {
            throw std::runtime_error("Tensor attributes must have non-empty dimensions and strides.");
        }
    
        size_t count = 1;
        for(const auto& dim : _dims)
        {
            count *= static_cast<size_t>(dim);
        }
    
        _memory.resize(count);
    }
    const std::vector<int64_t>& dims() const
    {
        return _dims;
    }

    const std::vector<int64_t>& strides() const
    {
        return _strides;
    }

    const Migratable_memory& memory() const
    {
        return _memory;
    }

    Migratable_memory& memory()
    {
        return _memory;
    }

private:
    Migratable_memory _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk