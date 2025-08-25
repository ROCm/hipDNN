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

// NOLINTBEGIN(portability-template-virtual-member-function)

template <typename T>
class Tensor_interface
{
public:
    virtual ~Tensor_interface() = default;

    virtual const std::vector<int64_t>& dims() const = 0;
    virtual const std::vector<int64_t>& strides() const = 0;

    virtual Migratable_memory_interface<T>& memory() = 0;
    virtual const Migratable_memory_interface<T>& memory() const = 0;

    virtual T get_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const = 0;
    virtual void set_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx, T value)
        = 0;
    virtual int64_t get_index(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const = 0;
    virtual void fill_with_value(T value) = 0;
    virtual void fill_with_random_values(T min, T max, unsigned int seed = std::random_device{}())
        = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

template <class T, class HostAlloc = Host_allocator<T>, class DeviceAlloc = Device_allocator<T>>
class Tensor : public Tensor_interface<T>
{
public:
    Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _memory(calculate_item_count(dims))
        , _dims(dims)
        , _strides(strides)
    {
        if(!is_packed(dims, strides))
        {
            throw std::invalid_argument("Tensor must be packed");
        }
    }

    Tensor(const std::vector<int64_t>& dims, const Tensor_layout& layout = Tensor_layout::NCHW)
        : Tensor(dims, generate_strides(dims, layout.stride_order))
    {
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    const std::vector<int64_t>& dims() const override
    {
        return _dims;
    }

    const std::vector<int64_t>& strides() const override
    {
        return _strides;
    }

    const Migratable_memory_interface<T>& memory() const override
    {
        return _memory;
    }

    Migratable_memory_interface<T>& memory() override
    {
        return _memory;
    }

    T get_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const override
    {
        int64_t index = get_index(nidx, cidx, hidx, widx);
        const auto* data = _memory.host_data();
        return data[index];
    }

    void set_host_value(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx, T value) override
    {
        int64_t index = get_index(nidx, cidx, hidx, widx);
        auto* data = _memory.host_data();
        data[index] = value;
    }

    int64_t get_index(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const override
    {
        return (nidx * _strides[0]) + (cidx * _strides[1]) + (hidx * _strides[2])
               + (widx * _strides[3]);
    }

    void fill_with_value(T value) override
    {
        T* data = _memory.host_data();
        std::fill(data, data + _memory.count(), value);
    }

    void fill_with_random_values(T min, T max, unsigned int seed = std::random_device{}()) override
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
    bool is_packed(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) const
    {
        // Item count = largest stride * item count in that dimension
        return (calculate_item_count(dims) == calculate_element_space(dims, strides));
    }

    static size_t calculate_element_space(const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& strides)
    {
        return static_cast<size_t>(
            std::inner_product(dims.begin(),
                               dims.end(),
                               strides.begin(),
                               1,
                               std::plus<>(),
                               [](size_t len, size_t stride) { return (len - 1) * stride; }));
    }

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

template <typename T>
using PinnedTensor = Tensor<T, Pinned_host_allocator<T>>;

} // namespace utilities
} // namespace hipdnn_sdk
