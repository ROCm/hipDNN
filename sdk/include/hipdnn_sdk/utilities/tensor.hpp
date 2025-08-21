// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/migratable_memory.hpp>
#include <hipdnn_sdk/utilities/shape_utils.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace hipdnn_sdk
{
namespace utilities
{

struct Tensor_layout
{
    std::string name;
    std::vector<int64_t> stride_order;

    static const Tensor_layout NCHW;
    static const Tensor_layout NHWC;
};

inline const Tensor_layout Tensor_layout::NCHW{.name = "NCHW", .stride_order = {3, 2, 1, 0}};
inline const Tensor_layout Tensor_layout::NHWC{.name = "NHWC",
                                               .stride_order = stride_order_nhwc(4)};

inline std::ostream& operator<<(std::ostream& os, const Tensor_layout& layout)
{
    return os << layout.name;
}

// Wraps vectors of dims/strides and Migratable_memory<T> to provide a common interface for testing
template <class T, class HostAlloc = Host_allocator<T>, class DeviceAlloc = Device_allocator<T>>
class Tensor
{
public:
    Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _memory(calculate_item_count(dims))
        , _dims(dims)
        , _strides(strides)
    {
    }

    Tensor(const std::vector<int64_t>& dims, const Tensor_layout& layout = Tensor_layout::NCHW)
        : _memory(calculate_item_count(dims))
        , _dims(dims)
        , _strides(generate_strides(dims, layout.stride_order))
    {
    }
    // Delete copy constructor and copy assignment operator

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Default move constructor and move assignment operator
    // These will automatically move _memory, _dims, and _strides
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    const std::vector<int64_t>& dims() const
    {
        return _dims;
    }

    const std::vector<int64_t>& strides() const
    {
        return _strides;
    }

    const Migratable_memory<T>& memory() const
    {
        return _memory;
    }

    Migratable_memory<T>& memory()
    {
        return _memory;
    }

    T get_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const
    {
        int64_t index = get_index(nidx, cidx, hidx, widx);
        const auto* data = memory().host_data();
        return data[index];
    }

    void set_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx, T value)
    {
        int64_t index = get_index(nidx, cidx, hidx, widx);
        auto* data = memory().host_data();
        data[index] = value;
    }

    int64_t get_index(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const
    {
        return (nidx * _strides[0]) + (cidx * _strides[1]) + (hidx * _strides[2])
               + (widx * _strides[3]);
    }

    void fill_with_value(T value)
    {
        T* data = _memory.host_data();
        std::fill(data, data + _memory.count(), value);
    }

    void fill_with_random_values(T min, T max, unsigned int seed = std::random_device{}())
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        auto* data = _memory.host_data();
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

    Migratable_memory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

} // namespace utilities
} // namespace hipdnn_sdk
