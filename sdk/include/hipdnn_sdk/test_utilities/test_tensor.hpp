// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <numeric>
#include <vector>

namespace hipdnn_sdk {
namespace reference_test_utilities {

using namespace hipdnn_sdk::utilities;

// Wraps vectors of dims/strides and Migratable_memory<T> to provide a common interface for testing
class Test_tensor
{
    private:
    Test_tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides, size_t item_size)
        : _memory(calculateItemCount(dims), item_size), _dims(dims), _strides(strides)
    {
    }

public:
    // Delete copy constructor and copy assignment operator
    Test_tensor(const Test_tensor&) = delete;
    Test_tensor& operator=(const Test_tensor&) = delete;
    
    // Default move constructor and move assignment operator
    // These will automatically move _memory, _dims, and _strides
    Test_tensor(Test_tensor&&) = default;
    Test_tensor& operator=(Test_tensor&&) = default;

    template<typename T>
    static Test_tensor make_test_tensor(const std::vector<int64_t>& dims, bool row_major = true)
    {
        return Test_tensor(dims, row_major ? calculateRowMajorStrides(dims) : calculateColumnMajorStrides(dims), sizeof(T));
    }

    template<typename T>
    static Test_tensor make_test_tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        return Test_tensor(dims, strides, sizeof(T));
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
    static size_t calculateItemCount(const std::vector<int64_t>& dims)
    {
        if (dims.empty()) {
            return 0;
        }

        return static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>()));
    }

    static std::vector<int64_t> calculateRowMajorStrides(const std::vector<int64_t>& dims) {
        size_t n = dims.size();
        std::vector<int64_t> strides(n, 1);
        if (n == 0) return strides;
        
        // Starting from the second-to-last dimension down to the first
        for (size_t i = n - 1; i-- > 0; ) {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
        
        return strides;
    }

    static std::vector<int64_t> calculateColumnMajorStrides(const std::vector<int64_t>& dims) {
        size_t n = dims.size();
        std::vector<int64_t> strides(n, 1);
        if (n == 0) return strides;
        
        // For column-major, we start from the second dimension and multiply by the size of the previous one.
        for (size_t i = 1; i < n; ++i) {
            strides[i] = dims[i - 1] * strides[i - 1];
        }
        
        return strides;
    }

    Migratable_memory _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk