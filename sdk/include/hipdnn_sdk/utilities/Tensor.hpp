// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/ShapeUtils.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace hipdnn_sdk
{
namespace utilities
{

struct TensorLayout
{
    std::string name;
    std::vector<int64_t> strideOrder;

    static const TensorLayout NCHW;
    static const TensorLayout NHWC;
};

inline const TensorLayout TensorLayout::NCHW{.name = "NCHW", .strideOrder = {3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NHWC{.name = "NHWC", .strideOrder = strideOrderNhwc(4)};

inline std::ostream& operator<<(std::ostream& os, const TensorLayout& layout)
{
    return os << layout.name;
}

// NOLINTBEGIN(portability-template-virtual-member-function)

template <typename T>
class ITensor
{
public:
    virtual ~ITensor() = default;

    virtual const std::vector<int64_t>& dims() const = 0;
    virtual const std::vector<int64_t>& strides() const = 0;

    virtual IMigratableMemory<T>& memory() = 0;
    virtual const IMigratableMemory<T>& memory() const = 0;

    virtual T getHostValue(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const = 0;
    virtual void setHostValue(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx, T value) = 0;
    virtual int64_t getIndex(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const = 0;
    virtual void fillWithValue(T value) = 0;
    virtual void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

template <class T, class HostAlloc = HostAllocator<T>, class DeviceAlloc = DeviceAllocator<T>>
class Tensor : public ITensor<T>
{
public:
    Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _memory(calculateItemCount(dims))
        , _dims(dims)
        , _strides(strides)
    {
        if(!isPacked(dims, strides))
        {
            throw std::invalid_argument("Tensor must be packed");
        }
    }

    Tensor(const std::vector<int64_t>& dims, const TensorLayout& layout = TensorLayout::NCHW)
        : Tensor(dims, generateStrides(dims, layout.strideOrder))
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

    const IMigratableMemory<T>& memory() const override
    {
        return _memory;
    }

    IMigratableMemory<T>& memory() override
    {
        return _memory;
    }

    T getHostValue(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const override
    {
        int64_t index = getIndex(nidx, cidx, hidx, widx);
        const auto* data = _memory.hostData();
        return data[index];
    }

    void setHostValue(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx, T value) override
    {
        int64_t index = getIndex(nidx, cidx, hidx, widx);
        auto* data = _memory.hostData();
        data[index] = value;
    }

    int64_t getIndex(int64_t nidx, int64_t cidx, int64_t hidx, int64_t widx) const override
    {
        return (nidx * _strides[0]) + (cidx * _strides[1]) + (hidx * _strides[2])
               + (widx * _strides[3]);
    }

    void fillWithValue(T value) override
    {
        T* data = _memory.hostData();
        std::fill(data, data + _memory.count(), value);
    }

    void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) override
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        auto* data = _memory.hostData();
        for(size_t i = 0; i < _memory.count(); ++i)
        {
            data[i] = static_cast<T>(distribution(generator));
        }
    }

private:
    bool isPacked(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) const
    {
        // Item count = largest stride * item count in that dimension
        return (calculateItemCount(dims) == calculateElementSpace(dims, strides));
    }

    static size_t calculateElementSpace(const std::vector<int64_t>& dims,
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

    static size_t calculateItemCount(const std::vector<int64_t>& dims)
    {
        if(dims.empty())
        {
            return 0;
        }

        return static_cast<size_t>(
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));
    }

    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

template <typename T>
using PinnedTensor = Tensor<T, PinnedHostAllocator<T>>;

} // namespace utilities
} // namespace hipdnn_sdk
