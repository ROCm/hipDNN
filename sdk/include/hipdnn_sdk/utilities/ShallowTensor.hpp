// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/ShapeUtils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk
{
namespace utilities
{

template <class T>
class ShallowHostOnlyMigratableMemory : public IMigratableMemory<T>
{
public:
    ShallowHostOnlyMigratableMemory(void* memory)
        : _memory(static_cast<T*>(memory))
    {
    }

    T* hostData() override
    {
        return _memory;
    }
    T* hostDataAsync() override
    {
        return _memory;
    }
    const T* hostData() const override
    {
        return _memory;
    }
    const T* hostDataAsync() const override
    {
        return _memory;
    }
    void* deviceData() override
    {
        throwNotSupported();
        return nullptr;
    }
    void* deviceDataAsync() override
    {
        throwNotSupported();
        return nullptr;
    }

    void markHostModified() override
    {
        //does nothing...
    }
    void markDeviceModified() override
    {
        throwNotSupported();
    }

    size_t count() const override
    {
        throwNotSupported();
        return 0;
    }
    bool empty() const override
    {
        throwNotSupported();
        return true;
    }
    MemoryLocation location() const override
    {
        return MemoryLocation::HOST;
    }

    void resize(size_t) override
    {
        throwNotSupported();
    }
    void clear() override
    {
        throwNotSupported();
    }

private:
    static void throwNotSupported()
    {
        throw std::runtime_error(
            "ShallowHostOnlyMigratableMemory only supports host data memory access. Resizes and "
            "allocations need to be done using MigratableMemeory.");
    }

    T* _memory;
};

template <class T>
class ShallowTensor : public TensorBase<T>
{
public:
    ShallowTensor(void* memory,
                  const std::vector<int64_t>& dims,
                  const std::vector<int64_t>& strides)
        : _memory(memory)
        , _dims(dims)
        , _strides(strides)
    {
    }

    ShallowTensor(const ShallowTensor&) = delete;
    ShallowTensor& operator=(const ShallowTensor&) = delete;

    ShallowTensor(ShallowTensor&&) = default;
    ShallowTensor& operator=(ShallowTensor&&) = default;

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

    void fillWithValue([[maybe_unused]] T value) override
    {
        //todo throw
        //noop, view only
    }

    void fillWithRandomValues([[maybe_unused]] T min,
                              [[maybe_unused]] T max,
                              [[maybe_unused]] unsigned int seed = std::random_device{}()) override
    {
        //todo throw
        //noop, view only
    }

private:
    ShallowHostOnlyMigratableMemory<T> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

}
}
