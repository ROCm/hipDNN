// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/shape_utils.hpp>
#include <numeric>
#include <random>
#include <vector>

namespace hipdnn_sdk
{
namespace utilities
{

// Wraps vectors of dims/strides and Migratable_memory<T> to provide a common interface for testing
class Tensor
{
private:
    Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides, size_t item_size)
        : _memory(calculate_item_count(dims), item_size)
        , _dims(dims)
        , _strides(strides)
    {
    }

public:
    // Delete copy constructor and copy assignment operator
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Default move constructor and move assignment operator
    // These will automatically move _memory, _dims, and _strides
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    template <typename T>
    static Tensor make_nchw_tensor(const std::vector<int64_t>& dims)
    {
        return {dims, generate_strides(dims, {3, 2, 1, 0}), sizeof(T)};
    }

    template <typename T>
    static Tensor make_tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        return {dims, strides, sizeof(T)};
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

    template <typename T>
    T get_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const
    {
        int64_t index = get_index(nidx, cidx, hidx, widx);
        const auto* data = memory().host_data<T>();
        return data[index];
    }

    template <typename T>
    void set_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx, T value)
    {
        int64_t index = get_index(nidx, cidx, hidx, widx);
        auto* data = memory().host_data<T>();
        data[index] = value;
    }

    int64_t get_index(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const
    {
        return (nidx * _strides[0]) + (cidx * _strides[1]) + (hidx * _strides[2])
               + (widx * _strides[3]);
    }

    template <typename T>
    void fill_with_value(T value)
    {
        T* data = _memory.host_data<T>();
        std::fill(data, data + _memory.count(), value);
    }

    template <typename T>
    void fill_with_random_values(T min, T max, unsigned int seed = std::random_device{}())
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        auto* data = _memory.host_data<T>();
        for(size_t i = 0; i < _memory.count(); ++i)
        {
            data[i] = static_cast<T>(distribution(generator));
        }
    }

private:
    static size_t calculate_item_count(const std::vector<int64_t>& dims)
    {
        if(dims.empty())
        {
            return 0;
        }

        return static_cast<size_t>(
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));
    }

    Migratable_memory _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

} // namespace reference_test_utilities
} // namespace hipdnn_sdk